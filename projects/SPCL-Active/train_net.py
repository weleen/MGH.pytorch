# encoding: utf-8
"""
@author:  wuyiming
@contact: yimingwu@hotmail.com
@function: Implementation of Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID
"""

import logging
import sys

sys.path.append('.')

from torch import nn
from fvcore.common.checkpoint import Checkpointer
from fvcore.nn.precise_bn import get_bn_modules

from fastreid.config import cfg
from fastreid.engine import default_argument_parser, default_setup, launch

from spclreid import *


def setup(args):
    """
    Create configs and perform basic setups.
    """
    add_spclreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    logger = logging.getLogger('fastreid.' + __name__)
    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = SPCLTrainer.build_model(cfg)
        model = nn.DataParallel(model).to(cfg.MODEL.DEVICE)
        Checkpointer(model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS)  # load trained model
        if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(model):
            prebn_cfg = cfg.clone()
            prebn_cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN
            prebn_cfg.DATASETS.NAMES = tuple([cfg.TEST.PRECISE_BN.DATASET])  # set dataset name for PreciseBN
            logger.info("Prepare precise BN dataset")
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                model,
                # Build a new data loader to not affect training
                SPCLTrainer.build_train_loader(prebn_cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            ).update_stats()
        res = SPCLTrainer.test(cfg, model)
        return res

    trainer = SPCLTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
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
