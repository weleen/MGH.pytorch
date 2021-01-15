# encoding: utf-8
"""
@author:  wuyiming
@contact: yimingwu@hotmail.com
"""
import logging
import time
import datetime
from collections import OrderedDict

from typing import List

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from fvcore.common.config import CfgNode
from fvcore.common.checkpoint import Checkpointer

from fastreid.data.build import DATASET_REGISTRY, fast_batch_collator
from fastreid.data.samplers import RandomMultipleGallerySampler
from fastreid.data.transforms import build_transforms
from fastreid.engine.defaults import DefaultTrainer
from fastreid.utils.logger import setup_logger, log_every_n_seconds
from fastreid.utils import comm
from fastreid.evaluation.evaluator import inference_context

from .hybrid_memory import HybridMemory
from . import hooks
from .common import CommDataset

__all__ = ["SPCLTrainer"]


class SPCLTrainer(DefaultTrainer):
    def __init__(self, cfg: CfgNode) -> None:
        logger = logging.getLogger('fastreid.' + __name__)
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()

        # build model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model.module)
        logger.info('Prepare training set')
        data_loader = self.construct_unsupervised_dataloader(is_train=False)
        cfg = self.auto_scale_hyperparams(cfg, data_loader)

        # create hybrid memory
        self.memory = HybridMemory(num_features=cfg.MODEL.BACKBONE.FEAT_DIM,
                                   num_samples=len(data_loader.dataset),
                                   temp=cfg.UNSUPERVISED.MEMORY_TEMP,
                                   momentum=cfg.UNSUPERVISED.MEMORY_MOMENTUM).to(cfg.MODEL.DEVICE)
        # initialize instance features
        logger.info("Initialize instance features in the hybrid memory")
        features, _ = self.extract_features(model, data_loader)
        features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(data_loader.dataset.img_items)], 0)
        self.memory.features = F.normalize(features, dim=1).to(cfg.MODEL.DEVICE)

        super(DefaultTrainer, self).__init__(model, data_loader, optimizer, cfg.SOLVER.FP16_ENABLED)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = Checkpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        if cfg.SOLVER.SWA.ENABLED:
            self.max_iter = cfg.SOLVER.MAX_EPOCH + cfg.SOLVER.SWA.ITER
        else:
            self.max_iter = cfg.SOLVER.MAX_EPOCH
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def construct_unsupervised_dataloader(self,
                                          datalist: List = None,
                                          is_train: bool = False,
                                          ) -> torch.utils.data.DataLoader:
        """
        :param datalist: dataset list. if dataset is None, load the dataset.train.
        :param is_train: build training transformation and sampler.
        :return:
        """
        transforms = build_transforms(self.cfg, is_train=is_train)

        if datalist is None:
            img_items = list()
            for d in self.cfg.DATASETS.NAMES:
                dataset = DATASET_REGISTRY.get(d)(combineall=self.cfg.DATASETS.COMBINEALL)
                dataset.show_train()
                img_items.extend(dataset.train)
        else:
            img_items = datalist

        img_items = sorted(img_items)
        data_set = CommDataset(img_items, transforms, relabel=True)

        num_workers = self.cfg.DATALOADER.NUM_WORKERS
        batch_size = self.cfg.SOLVER.IMS_PER_BATCH
        num_instance = self.cfg.DATALOADER.NUM_INSTANCE

        rmgs_flag = num_instance > 0
        if is_train and rmgs_flag and datalist is not None:
            data_sampler = RandomMultipleGallerySampler(img_items, num_instance)
        else:
            data_sampler = None

        data_loader = torch.utils.data.DataLoader(
            data_set,
            num_workers=num_workers,
            batch_size=batch_size,
            sampler=data_sampler,
            shuffle=(not rmgs_flag) and is_train,
            collate_fn=fast_batch_collator,
            pin_memory=True,
            drop_last=is_train
        )
        return data_loader

    def extract_features(self,
                         model: torch.nn.Module,
                         data_loader: torch.utils.data.DataLoader) -> (OrderedDict, OrderedDict):
        logger = logging.getLogger('fastreid.' + __name__)
        logger.info("Start inference on {} images".format(len(data_loader.dataset)))

        total = len(data_loader)
        num_warmup = min(5, total - 1)
        start_time = time.perf_counter()
        total_compute_time = 0

        features = OrderedDict()
        labels = OrderedDict()

        with inference_context(model), torch.no_grad():
            for idx, inputs in enumerate(data_loader):
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_compute_time = 0

                start_compute_time = time.perf_counter()
                outputs = model(inputs)
                for fname, out_feat, pid in zip(inputs['img_path'], outputs, inputs['targets']):
                    features[fname] = out_feat
                    labels[fname] = pid
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time

                idx += 1
                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                seconds_per_img = total_compute_time / iters_after_start
                if idx >= num_warmup * 2 or seconds_per_img > 30:
                    total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                    eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                    log_every_n_seconds(
                        logging.INFO,
                        "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                            idx + 1, total, seconds_per_img, str(eta)
                        ),
                        n=30,
                    )

        # Measure the time only for this worker (before the synchronization barrier)
        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        # NOTE this format is parsed by grep
        logger.info(
            "Total inference time: {} ({:.6f} s / img per device)".format(
                total_time_str, total_time / (total - num_warmup)
            )
        )
        total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
        logger.info(
            "Total inference pure compute time: {} ({:.6f} s / img per device)".format(
                total_compute_time_str, total_compute_time / (total - num_warmup)
            )
        )
        return features, labels

    def build_hooks(self) -> List:
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        logger = logging.getLogger('fastreid.' + __name__)
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN
        cfg.DATASETS.NAMES = tuple([cfg.TEST.PRECISE_BN.DATASET])  # set dataset name for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
        ]

        # cluster with DBSCAN
        ret.append(
            hooks.ClusterHook(
                cfg.UNSUPERVISED.EPS,
                cfg.UNSUPERVISED.EPS_GAP,
                cfg.UNSUPERVISED.CLUSTER_EPOCH,
                reset_opt=cfg.UNSUPERVISED.RESET_OPT
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
            ))
        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        # run writers in the end, so that evaluation metrics are written
        ret.append(hooks.PeriodicWriter(self.build_writers(), cfg.SOLVER.LOG_ITERS))

        return ret

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
