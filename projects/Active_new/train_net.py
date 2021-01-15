# encoding: utf-8
"""
@author:  wuyiming
@contact: yimingwu@hotmail.com
@function: Implementation of Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID
"""

import logging
import sys

sys.path.append('.')

import time
import itertools
import collections
import torch
import torch.nn.functional as F
from torch.cuda import amp
from torch import nn
from fvcore.common.checkpoint import Checkpointer

from fastreid.config import cfg
from fastreid.engine import default_argument_parser, default_setup, launch, hooks
from fastreid.engine.defaults import DefaultTrainer
from fastreid.data import build_reid_train_loader
from fastreid.utils.torch_utils import extract_features
from fastreid.utils import comm

from hooks import ActiveClusterHook, ActiveHook
from active_triplet_loss import ActiveTripletLoss
from fastreid.data.build import DATASET_REGISTRY, fast_batch_collator
from fastreid.data.common import CommDataset
from fastreid.data import samplers
from fastreid.data.transforms import build_transforms
import os

_root = os.getenv("FASTREID_DATASETS", "datasets")


class ActiveTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(ActiveTrainer, self).__init__(cfg)
        self.init_active()

    def init_active(self):
        logger = logging.getLogger('fastreid.' + __name__)
        logger.info('Build active dataloader')
        
        data_loader_active = self.build_active_sample_dataloader(is_train=False)
        self.data_loader_active = data_loader_active
        self._data_loader_active_iter = iter(data_loader_active)
        self.active_warmup = False
        self.active_loss = ActiveTripletLoss(self.cfg)

    def build_active_sample_dataloader(self, datalist=None, is_train=False):
        logger = logging.getLogger('fastreid.' + __name__)
        transforms = build_transforms(self.cfg, is_train=is_train)
        img_items = list()
        for d in self.cfg.DATASETS.NAMES:
            dataset = DATASET_REGISTRY.get(d)(root=_root, combineall=self.cfg.DATASETS.COMBINEALL)
            dataset.show_train()
            dataset.data.sort()
            img_items.append(dataset)

        data_set = CommDataset(img_items, transforms, relabel=True)

        num_workers = self.cfg.DATALOADER.NUM_WORKERS
        batch_size = self.cfg.ACTIVE.IMS_PER_BATCH
        num_instance = self.cfg.DATALOADER.NUM_INSTANCE

        if datalist is None:
            data_sampler = samplers.TrainingSampler(len(data_set))
        else:
            logger.info('Length of the triplet set: %d' % len(datalist))
            data_sampler = samplers.ActiveTripletSampler(datalist)

        batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, batch_size, drop_last=True)

        data_loader = torch.utils.data.DataLoader(
            data_set,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=fast_batch_collator,
        )
        
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

        if cfg.PSEUDO.ENABLED:
            assert len(cfg.PSEUDO.UNSUP ) > 0, "there are no dataset for unsupervised learning"
            ret.append(
                ActiveClusterHook(
                    self.cfg,
                    self.model,
                    len(self.data_loader.dataset)
                )
            )

        if cfg.ACTIVE.ENABLED:
            ret.append(
                ActiveHook(cfg, len(self.data_loader.dataset))
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

        def test_and_save_results(val=False):
            self._last_eval_results = self.test(self.cfg, self.model, val=val)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results, do_val=self.cfg.TEST.DO_VAL, metric_names=self.cfg.TEST.METRIC_NAMES))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), cfg.SOLVER.LOG_ITERS))

        return ret

    def run_step(self) -> None:
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SPCLTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        with amp.autocast(enabled=self.amp_enabled):
            # outs = self.model(data)

            # # Compute loss
            # if hasattr(self.model, 'module'):
            #     loss_dict = self.model.module.losses(outs, inputs=data)
            # else:
            #     loss_dict = self.model.losses(outs, inputs=data)
            loss_dict = {}

            if self.cfg.ACTIVE.ENABLED and self.active_warmup:
                data_active = next(self._data_loader_active_iter)
                outs_a = self.model(data_active)
                active_loss = ActiveTripletLoss(self.cfg)
                loss_dict.update({'active_loss': active_loss(outs_a)})

            losses = sum(loss_dict.values())

        with torch.cuda.stream(torch.cuda.Stream()):
            metrics_dict = loss_dict
            metrics_dict["data_time"] = data_time
            self._write_metrics(metrics_dict)
            self._detect_anomaly(losses, loss_dict)

        self.optimizer.zero_grad()

        if self.amp_enabled:
            self.scaler.scale(losses).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses.backward()
            self.optimizer.step()


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = ActiveTrainer.build_model(cfg)
        Checkpointer(model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS)  # load trained model
        res = ActiveTrainer.test(cfg, model)
        return res

    trainer = ActiveTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.test(trainer.cfg, trainer.model)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
