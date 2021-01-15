# encoding: utf-8
"""
@author:  tianjian
"""
import logging
import time

from typing import List

import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.parallel import DistributedDataParallel, DataParallel
from fvcore.common.config import CfgNode
from fvcore.common.checkpoint import Checkpointer

from fastreid.data.build import DATASET_REGISTRY, fast_batch_collator
from fastreid.data import samplers
from fastreid.data.transforms import build_transforms
from fastreid.engine.defaults import DefaultTrainer
from fastreid.utils.logger import setup_logger, log_every_n_seconds
from fastreid.utils import comm

from . import hooks
from .common import CommDataset

__all__ = ["HCTTrainer"]


class HCTTrainer(DefaultTrainer):
    def __init__(self, cfg: CfgNode) -> None:
        self.cfg = cfg
        logger = logging.getLogger('fastreid.' + __name__)
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()
        # build model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        logger.info('Prepare unlabeled training set')
        data_loader, self.train_set = self.build_hct_train_loader()
        cfg = self.auto_scale_hyperparams(cfg, data_loader)
        # For training, wrap with DP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,find_unused_parameters=True
            )
        else:
            model = DataParallel(model)

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
            self.max_iter = cfg.SOLVER.MAX_EPOCH + cfg.SOLVER.SWA.ITER
        else:
            self.max_iter = cfg.SOLVER.MAX_EPOCH
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def build_hct_train_loader(self, datalist=None):
        if datalist:
            data_set = datalist
        else:
            train_transforms = build_transforms(self.cfg, is_train=True)
            train_items = list()
            for d in self.cfg.DATASETS.NAMES:
                dataset = DATASET_REGISTRY.get(d)(combineall=self.cfg.DATASETS.COMBINEALL)
                dataset.show_train()
                train_items.extend(dataset.train)

            data_set = CommDataset(train_items, train_transforms, relabel=True)

        num_workers = self.cfg.DATALOADER.NUM_WORKERS
        batch_size = self.cfg.SOLVER.IMS_PER_BATCH
        num_instance = self.cfg.DATALOADER.NUM_INSTANCE

        if self.cfg.DATALOADER.PK_SAMPLER:
            data_sampler = samplers.NaiveIdentitySampler(data_set.img_items, batch_size, num_instance)
        else:
            data_sampler = samplers.TrainingSampler(len(data_set))
        batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, batch_size, True)

        data_loader = torch.utils.data.DataLoader(
            data_set,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=fast_batch_collator,
        )
        if datalist:
            return data_loader
        else:
            return data_loader, data_set

    def build_hooks(self) -> List:
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        logger = logging.getLogger('fastreid.' + __name__)
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN
        cfg.DATASETS.NAMES = tuple([cfg.TEST.PRECISE_BN.DATASET])  # set dataset name for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
        ]
        
        # HCT Hook
        ret.append(
            hooks.HCTHook(cfg, self.train_set)
        )

        if cfg.SOLVER.SWA.ENABLED:
            ret.append(
                hooks.SWA(
                    cfg.SOLVER.MAX_EPOCH,
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
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        # run writers in the end, so that evaluation metrics are written
        ret.append(hooks.PeriodicWriter(self.build_writers(), cfg.SOLVER.LOG_ITERS))

        return ret

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        cids = data['cid'].cuda()
        data_time = time.perf_counter() - start
        """
        If your want to do something with the heads, you can wrap the model.
        """
        outputs = self.model(data)

        if hasattr(self.model, 'module'):
            loss_dict = self.model.module.losses(outputs)
        else:
            loss_dict = self.model.losses(outputs) # ouputs->Tuple, len=3, 0->logits, 1->feats, 2->targets
        losses = sum(loss for loss in loss_dict.values())
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()

        self.optimizer.step()
