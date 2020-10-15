# encoding: utf-8

import sys

sys.path.append('.')

from torch import nn
from fvcore.common.checkpoint import Checkpointer

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

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = SPCLTrainer.build_model(cfg)
        Checkpointer(model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS)  # load trained model
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
