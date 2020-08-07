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

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            # ref to https://github.com/pytorch/pytorch/issues/22049 to set `find_unused_parameters=True`
            # for part of the parameters is not updated.
            ddp_cfg = {
                'device_ids': [comm.get_local_rank()],
                'broadcast_buffers': False,
                'output_device': comm.get_local_rank(),
                'find_unused_parameters': True
            }
            model = DistributedDataParallel(
                model, **ddp_cfg
            )

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