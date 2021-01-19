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

from hooks import RLCCLabelGeneratorHook
from hybrid_memory import RLCCMemory
from config import add_rlcc_config

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

        self.memory = RLCCMemory(num_features=cfg.MODEL.BACKBONE.FEAT_DIM,
                                 num_memory=num_memory,
                                 temp=cfg.PSEUDO.MEMORY.TEMP,
                                 momentum=cfg.PSEUDO.MEMORY.MOMENTUM,
                                 soft_label=cfg.PSEUDO.MEMORY.SOFT_LABEL,
                                 soft_label_start_epoch=cfg.PSEUDO.MEMORY.SOFT_LABEL_START_EPOCH,
                                 rlcc_start_epoch=cfg.PSEUDO.RLCC.START_EPOCH).to(cfg.MODEL.DEVICE)
        features, _ = extract_features(self.model, data_loader, norm_feat=self.cfg.PSEUDO.NORM_FEAT)#, save_path=os.path.join(self.cfg.OUTPUT_DIR, 'extract_features', '_'.join(self.cfg.DATASETS.NAMES)))
        datasets_size = data_loader.dataset.datasets_size
        datasets_size_range = list(itertools.accumulate([0] + datasets_size))
        memory_features = []
        memory_labels = []
        for idx, set in enumerate(common_dataset.datasets):
            start_id, end_id = datasets_size_range[idx], datasets_size_range[idx+1]
            assert end_id - start_id == len(set)
            if idx in self.cfg.PSEUDO.UNSUP:
                # init memory for unlabeled dataset with instance features
                memory_features.append(features[start_id: end_id])
                labels = torch.arange(start_id, end_id)
                memory_labels.append(labels)
            else:
                # init memory for labeled dataset with class center features
                centers_dict = collections.defaultdict(list)
                labels = []
                for i, (_, pid, _) in enumerate(set.data):
                    centers_dict[common_dataset.pid_dict[pid]].append(features[i].unsqueeze(0))
                    labels.append(common_dataset.pid_dict[pid])
                centers = [
                    torch.cat(centers_dict[pid], 0).mean(0) for pid in sorted(centers_dict.keys())
                ]
                memory_features.append(torch.stack(centers, 0))
                memory_labels.append(torch.stack(labels, 0))

        self.memory._update_feature(torch.cat(memory_features))
        self.memory._update_label(torch.cat(memory_labels))
        del data_loader, common_dataset, features
        
    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN
        cfg.DATASETS.NAMES = tuple([cfg.TEST.PRECISE_BN.DATASET])  # set dataset name for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
        ]

        if cfg.PSEUDO.ENABLED:
            assert len(cfg.PSEUDO.UNSUP) > 0, "there are no dataset for unsupervised learning"
            ret.append(
                RLCCLabelGeneratorHook(
                    self.cfg,
                    self.model
                )
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
            ret.append(hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            ))

        if cfg.MODEL.OPEN_LAYERS != [''] and cfg.SOLVER.FREEZE_ITERS > 0:
            open_layers = ",".join(cfg.MODEL.OPEN_LAYERS)
            self._logger.info(f'Open "{open_layers}" training for {cfg.SOLVER.FREEZE_ITERS:d} iters')
            ret.append(hooks.FreezeLayer(
                self.model,
                cfg.MODEL.OPEN_LAYERS,
                cfg.SOLVER.FREEZE_ITERS,
                cfg.SOLVER.FREEZE_FC_ITERS,
            ))
        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD * self.iters_per_epoch, self.max_iter))

        def test_and_save_results(mode='test'):
            self._last_eval_results = self.test(self.cfg, self.model, mode=mode)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results, do_val=self.cfg.TEST.DO_VAL, metric_names=self.cfg.TEST.METRIC_NAMES))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), cfg.SOLVER.LOG_ITERS))

        return ret

    def run_step(self) -> None:
        assert self.model.training, "[SpCLTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        outs = self.model(data)

        # Compute loss
        if hasattr(self.model, 'module'):
            loss_dict = self.model.module.losses(outs, memory=self.memory, inputs=data)
        else:
            loss_dict = self.model.losses(outs, memory=self.memory, inputs=data)

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
    add_rlcc_config(cfg)
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
