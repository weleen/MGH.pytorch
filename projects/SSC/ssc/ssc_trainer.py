# encoding: utf-8
"""
@author:  wuyiming
@contact: yimingwu@hotmail.com
"""
import logging
import time
import datetime
from collections import OrderedDict

import torch
from torch.nn.parallel import DistributedDataParallel
from fvcore.common.config import CfgNode
from fvcore.common.checkpoint import Checkpointer

from fastreid.engine.defaults import DefaultTrainer
from fastreid.utils.logger import setup_logger
from fastreid.utils import comm

from .data.build import build_reid_train_loader


__all__ = ["SSCTrainer"]


class SSCTrainer(DefaultTrainer):
    def __init__(self, cfg: CfgNode) -> None:
        logger = logging.getLogger('fastreid.' + __name__)
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for fastreid
            setup_logger()

        # Assume these objects must be constructed in this order.
        data_loader = self.build_train_loader(cfg)
        cfg = self.auto_scale_hyperparams(cfg, data_loader)
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        super(DefaultTrainer, self).__init__(model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = Checkpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            save_to_disk=comm.is_main_process(),
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

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It now calls :func:`fastreid.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        logger = logging.getLogger(__name__)
        logger.info("Prepare training set")
        return build_reid_train_loader(cfg)

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
        data_time = time.perf_counter() - start

        """
        If your want to do something with the heads, you can wrap the model.
        """
        outputs = self.model(data)
        if hasattr(self.model, 'module'):
            loss_dict = self.model.module.losses(outputs, self.iter)
        else:
            loss_dict = self.model.losses(outputs, self.iter)
        losses = sum(loss_dict.values())
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        """
        If you need accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        self.optimizer.step()

    @staticmethod
    def auto_scale_hyperparams(cfg, data_loader):
        r"""
        This is used for auto-computation actual training iterations,
        because some hyper-param, such as MAX_ITER, means training epochs rather than iters,
        so we need to convert specific hyper-param to training iterations.
        """

        cfg = cfg.clone()
        frozen = cfg.is_frozen()
        cfg.defrost()

        iters_per_epoch = len(data_loader.dataset) // cfg.SOLVER.IMS_PER_BATCH
        cfg.DATALOADER.ITERS_PER_EPOCH = iters_per_epoch
        cfg.MODEL.HEADS.NUM_CLASSES = data_loader.dataset.num_classes
        cfg.SOLVER.MAX_ITER *= iters_per_epoch
        cfg.SOLVER.WARMUP_ITERS *= iters_per_epoch
        cfg.SOLVER.FREEZE_ITERS *= iters_per_epoch
        cfg.SOLVER.DELAY_ITERS *= iters_per_epoch
        for i in range(len(cfg.SOLVER.STEPS)):
            cfg.SOLVER.STEPS[i] *= iters_per_epoch
        cfg.SOLVER.SWA.ITER *= iters_per_epoch
        cfg.SOLVER.SWA.PERIOD *= iters_per_epoch
        cfg.UNSUPERVISED.RAMPUP_ITER *= iters_per_epoch

        # Evaluation period must be divided by cfg.SOLVER.LOG_PERIOD for writing into tensorboard.
        num_mode = cfg.SOLVER.LOG_PERIOD - (cfg.TEST.EVAL_PERIOD * iters_per_epoch) % cfg.SOLVER.LOG_PERIOD
        cfg.TEST.EVAL_PERIOD = cfg.TEST.EVAL_PERIOD * iters_per_epoch + num_mode

        num_mode = cfg.SOLVER.LOG_PERIOD - (cfg.SOLVER.CHECKPOINT_PERIOD * iters_per_epoch) % cfg.SOLVER.LOG_PERIOD
        cfg.SOLVER.CHECKPOINT_PERIOD = cfg.SOLVER.CHECKPOINT_PERIOD * iters_per_epoch + num_mode

        logger = logging.getLogger(__name__)
        logger.info(
            f"Auto-scaling the config to num_classes={cfg.MODEL.HEADS.NUM_CLASSES}, "
            f"max_Iter={cfg.SOLVER.MAX_ITER}, wamrup_Iter={cfg.SOLVER.WARMUP_ITERS}, "
            f"freeze_Iter={cfg.SOLVER.FREEZE_ITERS}, delay_Iter={cfg.SOLVER.DELAY_ITERS}, "
            f"step_Iter={cfg.SOLVER.STEPS}, ckpt_Iter={cfg.SOLVER.CHECKPOINT_PERIOD}, "
            f"eval_Iter={cfg.TEST.EVAL_PERIOD}, "
            f"rampup_Iter={cfg.UNSUPERVISED.RAMPUP_ITER}, "
            f"iters_per_epoch={iters_per_epoch}."
        )

        if frozen: cfg.freeze()

        return cfg