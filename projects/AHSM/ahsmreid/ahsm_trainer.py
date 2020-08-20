# encoding: utf-8
"""
@author:  wenhuzhang
@contact: Andrew-pph@outlook.com
"""
import logging
import time
import datetime
from collections import OrderedDict

from typing import List

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel, DataParallel
from fvcore.common.config import CfgNode
from fvcore.common.checkpoint import Checkpointer
from torch.utils.data.sampler import SubsetRandomSampler

from fastreid.data.build import DATASET_REGISTRY, fast_batch_collator

from fastreid.data.transforms import build_transforms
from fastreid.engine.defaults import DefaultTrainer
from fastreid.utils.logger import setup_logger, log_every_n_seconds
from fastreid.utils import comm
from fastreid.evaluation.evaluator import inference_context

# from fastreid.engine import hooks
from fastreid.data.common import CommDataset

import random
from fastreid.data import samplers

from . import hooks

__all__ = ["AHSMTrainer"]


class AHSMTrainer(DefaultTrainer):
    def __init__(self, cfg: CfgNode) -> None:
        self.cfg = cfg
        logger = logging.getLogger('fastreid.' + __name__)
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()
        # build model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        logger.info('Prepare active hard sampling data set')
        data_loader, self.data_len = self.build_active_sample_dataloader(is_train=False)       
        # For training, wrap with DP. But don't need this for inference.
        model = DataParallel(model)
        model = model.cuda()
        super(DefaultTrainer, self).__init__(model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = Checkpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        if cfg.SOLVER.SWA.ENABLED:
            self.max_iter = cfg.SOLVER.MAX_ITER + cfg.SOLVER.SWA.ITER
        else:
            self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        
        self.register_hooks(self.build_hooks())

    def build_active_sample_dataloader(self, datalist: set = None, is_train: bool = False, is_sample_loader:bool=False) -> torch.utils.data.DataLoader:
        """
        :param datalist: dataset list. if dataset is None, random initialize the labeled/unlabeled list.
        :param is_train: build training transformation and sampler.
        :return:
        """
        transforms = build_transforms(self.cfg, is_train=is_train)
        train_items = list()
        for d in self.cfg.DATASETS.NAMES:
            dataset = DATASET_REGISTRY.get(d)(combineall=self.cfg.DATASETS.COMBINEALL)
            dataset.show_train()
            train_items.extend(dataset.train)

        data_set = CommDataset(train_items, transforms, relabel=True)
        if datalist: 
            print('Length of the triplet set:', len(datalist))
        num_workers = self.cfg.DATALOADER.NUM_WORKERS
        batch_size = self.cfg.SOLVER.IMS_PER_BATCH
        num_instance = self.cfg.DATALOADER.NUM_INSTANCE

        if self.cfg.DATALOADER.PK_SAMPLER:
            data_sampler = samplers.RandomIdentitySampler(data_set.img_items, batch_size, num_instance)
        else:
            data_sampler = samplers.TrainingSampler(len(data_set))

        if datalist is None:
            data_len = len(train_items)
        else:
            data_sampler = samplers.ActiveTripletSampler(datalist)

        batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, batch_size, True)

        data_loader = torch.utils.data.DataLoader(
            data_set,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=fast_batch_collator,
        )
        if datalist is None:
            return data_loader, data_len
        else:
            return data_loader

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        logger = logging.getLogger(__name__)
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN
        cfg.DATASETS.NAMES = tuple([cfg.TEST.PRECISE_BN.DATASET])  # set dataset name for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
        ]

        # Dataloader Hook
        ret.append(
            hooks.DataloaderHook(cfg, self.data_len)
        )

        if cfg.SOLVER.SWA.ENABLED:
            ret.append(
                hooks.SWA(
                    cfg.SOLVER.MAX_ITER,
                    cfg.SOLVER.SWA.PERIOD,
                    cfg.SOLVER.SWA.LR_FACTOR,
                    cfg.SOLVER.SWA.ETA_MIN_LR,
                    cfg.SOLVER.SWA.LR_SCHED,
                )
            )

        if cfg.TEST.PRECISE_BN.ENABLED and hooks.get_bn_modules(self.model):
            logger.info("Prepare precise BN dataset")
            ret.append(hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            ))

        if cfg.MODEL.OPEN_LAYERS != [''] and cfg.SOLVER.FREEZE_ITERS > 0:
            open_layers = ",".join(cfg.MODEL.OPEN_LAYERS)
            logger.info(f'Open "{open_layers}" training for {cfg.SOLVER.FREEZE_ITERS:d} iters')
            ret.append(hooks.FreezeLayer(
                self.model,
                cfg.MODEL.OPEN_LAYERS,
                cfg.SOLVER.FREEZE_ITERS,
            ))
        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        # if comm.is_main_process():
        ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        # run writers in the end, so that evaluation metrics are written
        ret.append(hooks.PeriodicWriter(self.build_writers(), cfg.SOLVER.LOG_PERIOD))
        return ret
    



