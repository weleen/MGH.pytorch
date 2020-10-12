'''
Author: your name
Date: 2020-10-10 22:24:22
LastEditTime: 2020-10-12 17:28:32
LastEditors: your name
Description: In User Settings Edit
FilePath: /git/fast-reid/projects/CompactReID/train_net.py
'''
# encoding: utf-8
"""
@author:  wuyiming
"""

import sys

sys.path.append('.')

from torch.nn.parallel import DistributedDataParallel
from fastreid.config import cfg
from fastreid.engine import default_argument_parser, default_setup, launch
from fvcore.common.checkpoint import Checkpointer
from fastreid.utils import comm
from compactreid import *


def setup(args):
    """
    Create configs and perform basic setups.
    """
    add_compactreid_config(cfg)
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
        model = CompactTrainer.build_model(cfg)
        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = CompactTrainer.test(cfg, model)
        return res

    trainer = CompactTrainer(cfg)
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
