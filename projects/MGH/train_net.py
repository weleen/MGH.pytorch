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
from fvcore.common.checkpoint import Checkpointer

import torch
from fastreid.config import cfg
from fastreid.engine import default_argument_parser, default_setup, launch, hooks
from fastreid.engine.defaults import DefaultTrainer
from fastreid.utils import comm
from fastreid.engine import hooks

from cap_memory import CAPMemory
from unified_memory import UnifiedMemory
from cap_labelgenerator import CAPLabelGeneratorHook
from config import add_cap_config
from instance_loss import instance_loss
from get_st_matrix import get_st_distribution
import numpy as np
from cluster import Cluster

try:
    import apex
    from apex import amp
    from apex.parallel import DistributedDataParallel
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example if you want to"
                      "train with DDP")

class CAPTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(CAPTrainer, self).__init__(cfg)

    def init_memory(self):
        logger = logging.getLogger('fastreid.' + __name__)
        logger.info("Initialize CAP memory")

        self.memory = UnifiedMemory(self.cfg)

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
            assert len(cfg.PSEUDO.UNSUP) > 0, "there are no dataset for unsupervised learning"
            ret.append(
                CAPLabelGeneratorHook(
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
        assert self.model.training, "[CAPTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        data = self.batch_process(data, is_dsbn=self.cfg.MODEL.DSBN)
        if self.cfg.MODEL.MEAN_NET:
            outs, outs_mean = self.model(data)
        else:
            outs = self.model(data)

        # Compute loss
        with torch.autograd.set_detect_anomaly(True):
            if self.cfg.MODEL.MEAN_NET:
                if hasattr(self.model, 'module'):
                    loss_dict = self.model.module.losses(outs, outs_mean=outs_mean, memory=self.memory, inputs=data)
                else:
                    loss_dict = self.model.losses(outs, outs_mean=outs_mean, memory=self.memory, inputs=data)
            else:
                if hasattr(self.model, 'module'):
                    loss_dict = self.model.module.losses(outs, memory=self.memory, inputs=data)
                else:
                    loss_dict = self.model.losses(outs, memory=self.memory, inputs=data)
        
        if self.cfg.CAP.INSTANCE_LOSS:
            un_data = next(self.un_data_loader_iter)
            un_data = self.batch_process(un_data, is_dsbn=self.cfg.MODEL.DSBN)
            un_outs = self.model(un_data)
            loss_dict.update(instance_loss(un_outs))

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
    add_cap_config(cfg)
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
        model = CAPTrainer.build_model(cfg)
        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model
        model = CAPTrainer.build_parallel_model(cfg, model)  # parallel
        
        if cfg.CAP.ST_TEST:
            save_path = os.path.join(cfg.OUTPUT_DIR, 'distribution.npy')
            if not os.path.exists(save_path):
                cluster = Cluster(cfg, model)
                all_labels, all_centers, all_features, all_camids = cluster.update_labels()
                items = cluster._data_loader_cluster.dataset.img_items
                imgs_paths = [items[i][0] for i, lb in enumerate(all_labels[0]) if lb != -1]
                pseudo_labels = [lb for lb in all_labels[0] if lb != -1]
                distribution = get_st_distribution(imgs_paths, cfg.DATASETS.NAMES[0],  pseudo_labels=pseudo_labels)
                np.save(save_path, distribution)

        res = CAPTrainer.test(cfg, model)
        return res

    trainer = CAPTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.init_memory()

    # items = trainer.data_loader.dataset.img_items
    # imgs_paths = [i[0] for i in items if i[1] != -1]
    # pseudo_labels = [i[1] for i in items if i[1] != -1]
    # distribution = get_st_distribution(imgs_paths, pseudo_labels=pseudo_labels)
    # np.save('train_distribution.npy', distribution)
    # trainer.test(trainer.cfg, trainer.model)

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
