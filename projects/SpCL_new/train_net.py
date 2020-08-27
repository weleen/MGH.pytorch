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
from torch import nn
from fvcore.common.checkpoint import Checkpointer

from fastreid.config import cfg
from fastreid.engine import default_argument_parser, default_setup, launch
from fastreid.engine.defaults import DefaultTrainer
from fastreid.data import build_reid_train_loader_new
from fastreid.utils.torch_utils import extract_features

from hybrid_memory import HybridMemory


class SPCLTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(SPCLTrainer, self).__init__(cfg)
        self.init_memory()

    def init_memory(self):
        logger = logging.getLogger('fastreid.' + __name__)
        logger.info("Initialize instance features in the hybrid memory")
        # build dataloader
        data_loader = build_reid_train_loader_new(self.cfg, is_train=False)
        common_dataset = data_loader.dataset
        # initialize memory features
        num_samples = 0
        for idx, set in enumerate(common_dataset.datasets):
            if idx in self.cfg.PSEUDO.UNSUP:
                num_samples += len(set)
            else:
                num_samples += set.num_train_pids

        self.memory = HybridMemory(num_features=cfg.MODEL.HEADS.IN_FEAT,
                                   num_samples=num_samples,
                                   temp=cfg.PSEUDO.MEMORY.TEMP,
                                   momentum=cfg.PSEUDO.MEMORY.MOMENTUM).to(cfg.MODEL.DEVICE)
        features, true_labels = extract_features(self.model, data_loader, norm_feat=self.cfg.PSEUDO.NORM_FEAT)
        datasets_size = self.data_loader.dataset.datasets_size
        datasets_size_range = list(itertools.accumulate([0] + datasets_size))
        memory_features = []
        for idx, set in enumerate(common_dataset.datasets):
            start_id, end_id = datasets_size_range[idx], datasets_size_range[idx+1]
            assert end_id - start_id == len(set)
            if idx in self.cfg.PSEUDO.UNSUP:
                memory_features.append(features[start_id: end_id])
            else:
                centers_dict = collections.defaultdict(list)
                for i, (_, pid, _) in enumerate(set.data):
                    centers_dict[common_dataset.pid_dict[pid]].append(features[i].unsqueeze(0))
                centers = [
                    torch.cat(centers_dict[pid], 0).mean(0) for pid in sorted(centers_dict.keys())
                ]
                memory_features.append(torch.stack(centers, 0))

        self.memory._update_feature(torch.cat(memory_features))

    def run_step(self) -> None:
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SPCLTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        outputs = self.model(data)
        if hasattr(self.model, 'module'):
            loss_dict = self.model.module.losses(outputs, self.memory)
        else:
            loss_dict = self.model.losses(outputs, self.memory)
        losses = sum(loss for loss in loss_dict.values())
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
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
        model = SPCLTrainer.build_model(cfg)
        model = nn.DataParallel(model).to(cfg.MODEL.DEVICE)
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
