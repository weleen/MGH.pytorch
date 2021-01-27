# encoding: utf-8
"""
@author:  wuyiming
@contact: yimingwu@hotmail.com
@function: Implementation of Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID
"""
import os
import logging
import sys

sys.path.append('.')

import time
import itertools
import collections
import torch
import torch.nn.functional as F
from torch import nn
from fvcore.common.checkpoint import Checkpointer

from fastreid.config import cfg
from fastreid.engine import default_argument_parser, default_setup, launch, hooks
from fastreid.engine.defaults import DefaultTrainer
from fastreid.data import build_reid_train_loader
from fastreid.utils.torch_utils import extract_features
from fastreid.utils import comm
from fastreid.modeling.losses.hybrid_memory import HybridMemory

try:
    import apex
    from apex import amp
    from apex.parallel import DistributedDataParallel
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example if you want to"
                      "train with DDP")

class SPCLTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(SPCLTrainer, self).__init__(cfg)
        # add weight_matrix for loss calculation
        self.weight_matrix = None

    def init_memory(self):
        logger = logging.getLogger('fastreid.' + __name__)
        logger.info("Initialize instance features in the hybrid memory")
        # initialize memory features
        logger.info("Build dataloader for initializing memory features")
        data_loader = build_reid_train_loader(self.cfg, is_train=False)
        common_dataset = data_loader.dataset

        num_memory = 0
        for idx, set in enumerate(common_dataset.datasets):
            if idx in self.cfg.PSEUDO.UNSUP:
                num_memory += len(set)
            else:
                num_memory += set.num_train_pids

        self.memory = HybridMemory(num_features=cfg.MODEL.BACKBONE.FEAT_DIM,
                                   num_memory=num_memory,
                                   temp=cfg.PSEUDO.MEMORY.TEMP,
                                   momentum=cfg.PSEUDO.MEMORY.MOMENTUM,
                                   weighted=cfg.PSEUDO.MEMORY.WEIGHTED,
                                   weight_mask_topk=cfg.PSEUDO.MEMORY.WEIGHT_MASK_TOPK,
                                   soft_label=cfg.PSEUDO.MEMORY.SOFT_LABEL,
                                   soft_label_start_epoch=cfg.PSEUDO.MEMORY.SOFT_LABEL_START_EPOCH).to(cfg.MODEL.DEVICE)
        features, _, _ = extract_features(self.model, data_loader, norm_feat=self.cfg.PSEUDO.NORM_FEAT, save_path=os.path.join(self.cfg.OUTPUT_DIR, 'extract_features', '_'.join(self.cfg.DATASETS.NAMES)))
        datasets_size = data_loader.dataset.datasets_size
        datasets_size_range = list(itertools.accumulate([0] + datasets_size))
        memory_features = []
        for idx, set in enumerate(common_dataset.datasets):
            start_id, end_id = datasets_size_range[idx], datasets_size_range[idx+1]
            assert end_id - start_id == len(set)
            if idx in self.cfg.PSEUDO.UNSUP:
                # init memory for unlabeled dataset with instance features
                memory_features.append(features[start_id: end_id])
            else:
                # init memory for labeled dataset with class center features
                centers_dict = collections.defaultdict(list)
                for i, (_, pid, _) in enumerate(set.data):
                    centers_dict[pid].append(features[i + start_id].unsqueeze(0))
                centers = [
                    torch.cat(centers_dict[pid], 0).mean(0) for pid in sorted(centers_dict.keys())
                ]
                memory_features.append(torch.stack(centers, 0))

        self.memory._update_feature(torch.cat(memory_features))
        del data_loader, common_dataset, features

    def run_step(self) -> None:
        assert self.model.training, "[SpCLTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        data = self.batch_process(data, is_dsbn=self.cfg.MODEL.DSBN)
        outs = self.model(data)

        # Compute loss
        if hasattr(self.model, 'module'):
            loss_dict = self.model.module.losses(outs, memory=self.memory, inputs=data, weight=self.weight_matrix)
        else:
            loss_dict = self.model.losses(outs, memory=self.memory, inputs=data, weight=self.weight_matrix)

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
        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model
        model = SPCLTrainer.build_parallel_model(cfg, model)  # parallel
        
        res = SPCLTrainer.test(cfg, model)
        return res

    trainer = SPCLTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.init_memory()
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
