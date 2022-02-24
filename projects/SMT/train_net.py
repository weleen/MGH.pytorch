# encoding: utf-8
"""
@author:  wuyiming
@contact: yimingwu@hotmail.com
@function: Implementation of SMT
"""
import sys

sys.path.append('.')

import time
from fvcore.common.checkpoint import Checkpointer

from fastreid.config import cfg
from fastreid.engine import default_argument_parser, default_setup, launch
from fastreid.engine.defaults import DefaultTrainer

try:
    import apex
    from apex import amp
    from apex.parallel import DistributedDataParallel
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example if you want to"
                      "train with DDP")

from config import add_smtreid_config


class SMTTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(SMTTrainer, self).__init__(cfg)

    def run_step(self) -> None:
        assert self.model.training, "[SMTTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        outs, outs_mean = self.model(data)

        # Compute loss
        if hasattr(self.model, 'module'):
            loss_dict = self.model.module.losses(outs, outs_mean=outs_mean)
        else:
            loss_dict = self.model.losses(outs, outs_mean=outs_mean)

        losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        if isinstance(self.model, DistributedDataParallel):  # if model is apex.DistributedDataParallel
            with amp.scale_loss(losses, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            losses.backward()

        self._write_metrics(loss_dict, data_time)

        self.optimizer.step()


def setup(args):
    """
    Create configs and perform basic setups.
    """
    add_smtreid_config(cfg)
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
        model = SMTTrainer.build_model(cfg)
        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model
        model = SMTTrainer.build_parallel_model(cfg, model)  # parallel

        res = SMTTrainer.test(cfg, model)
        return res

    trainer = SMTTrainer(cfg)
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
