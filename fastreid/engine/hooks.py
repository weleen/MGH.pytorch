# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import datetime
import itertools
import logging
import os
import tempfile
import time
from collections import Counter
import copy
import numpy as np
import collections

import torch
import torch.nn.functional as F
from torch import nn

from fastreid.evaluation.testing import flatten_results_dict
from fastreid.solver import optim
from fastreid.utils import comm
from fvcore.common.checkpoint import PeriodicCheckpointer as _PeriodicCheckpointer
from fastreid.utils.events import EventStorage, EventWriter, get_event_storage
from fvcore.common.file_io import PathManager
from fvcore.nn.precise_bn import update_bn_stats, get_bn_modules
from fvcore.common.timer import Timer
from fastreid.data import build_reid_train_loader
from fastreid.utils.clustering import label_generator_dbscan, label_generator_hypergraph, label_generator_kmeans, label_generator_cdp, label_generator_hypergraph_new
from fastreid.utils.metrics import cluster_metrics
from fastreid.utils.torch_utils import extract_features
from .train_loop import HookBase

__all__ = [
    "CallbackHook",
    "IterationTimer",
    "PeriodicWriter",
    "PeriodicCheckpointer",
    "LRScheduler",
    "AutogradProfiler",
    "EvalHook",
    "PreciseBN",
    "FreezeLayer",
    "LabelGeneratorHook"
]

"""
Implement some common hooks.
"""


class CallbackHook(HookBase):
    """
    Create a hook using callback functions provided by the user.
    """

    def __init__(self, *, before_train=None, after_train=None, before_epoch=None, after_epoch=None,
                 before_step=None, after_step=None):
        """
        Each argument is a function that takes one argument: the trainer.
        """
        self._before_train = before_train
        self._before_epoch = before_epoch
        self._before_step = before_step
        self._after_step = after_step
        self._after_epoch = after_epoch
        self._after_train = after_train

    def before_train(self):
        if self._before_train:
            self._before_train(self.trainer)

    def after_train(self):
        if self._after_train:
            self._after_train(self.trainer)
        # The functions may be closures that hold reference to the trainer
        # Therefore, delete them to avoid circular reference.
        del self._before_train, self._after_train
        del self._before_step, self._after_step

    def before_epoch(self):
        if self._before_epoch:
            self._before_epoch(self.trainer)

    def after_epoch(self):
        if self._after_epoch:
            self._after_epoch(self.trainer)

    def before_step(self):
        if self._before_step:
            self._before_step(self.trainer)

    def after_step(self):
        if self._after_step:
            self._after_step(self.trainer)


class IterationTimer(HookBase):
    """
    Track the time spent for each iteration (each run_step call in the trainer).
    Print a summary in the end of training.
    This hook uses the time between the call to its :meth:`before_step`
    and :meth:`after_step` methods.
    Under the convention that :meth:`before_step` of all hooks should only
    take negligible amount of time, the :class:`IterationTimer` hook should be
    placed at the beginning of the list of hooks to obtain accurate timing.
    """

    def __init__(self, warmup_iter=3):
        """
        Args:
            warmup_iter (int): the number of iterations at the beginning to exclude
                from timing.
        """
        self._warmup_iter = warmup_iter
        self._step_timer = Timer()

    def before_train(self):
        self._start_time = time.perf_counter()
        self._total_timer = Timer()
        self._total_timer.pause()

    def after_train(self):
        logger = logging.getLogger(__name__)
        total_time = time.perf_counter() - self._start_time
        total_time_minus_hooks = self._total_timer.seconds()
        hook_time = total_time - total_time_minus_hooks

        num_iter = self.trainer.iter + 1 - self.trainer.start_iter - self._warmup_iter

        if num_iter > 0 and total_time_minus_hooks > 0:
            # Speed is meaningful only after warmup
            # NOTE this format is parsed by grep in some scripts
            logger.info(
                "Overall training speed: {} iterations in {} ({:.4f} s / it)".format(
                    num_iter,
                    str(datetime.timedelta(seconds=int(total_time_minus_hooks))),
                    total_time_minus_hooks / num_iter,
                )
            )

        logger.info(
            "Total training time: {} ({} on hooks)".format(
                str(datetime.timedelta(seconds=int(total_time))),
                str(datetime.timedelta(seconds=int(hook_time))),
            )
        )

    def before_step(self):
        self._step_timer.reset()
        self._total_timer.resume()

    def after_step(self):
        # +1 because we're in after_step
        iter_done = self.trainer.iter - self.trainer.start_iter + 1
        if iter_done >= self._warmup_iter:
            sec = self._step_timer.seconds()
            self.trainer.storage.put_scalars(time=sec)
        else:
            self._start_time = time.perf_counter()
            self._total_timer.reset()

        self._total_timer.pause()


class PeriodicWriter(HookBase):
    """
    Write events to EventStorage periodically.
    It is executed every ``period`` iterations and after the last iteration.
    """

    def __init__(self, writers, period=20):
        """
        Args:
            writers (list[EventWriter]): a list of EventWriter objects
            period (int):
        """
        self._writers = writers
        for w in writers:
            assert isinstance(w, EventWriter), w
        self._period = period

    def after_step(self):
        if (self.trainer.iter + 1) % self._period == 0 or (
                self.trainer.iter == self.trainer.max_iter - 1
        ):
            for writer in self._writers:
                writer.write()

    def after_epoch(self):
        for writer in self._writers:
            writer.write()

    def after_train(self):
        for writer in self._writers:
            writer.close()


class PeriodicCheckpointer(_PeriodicCheckpointer, HookBase):
    """
    Same as :class:`fvcore.common.checkpoint.PeriodicCheckpointer`, but as a hook.
    Note that when used as a hook,
    it is unable to save additional data other than what's defined
    by the given `checkpointer`.
    It is executed every ``period`` iterations and after the last iteration.
    """

    def before_train(self):
        if len(self.trainer.cfg.DATASETS.TESTS) == 1:
            self.metric_name = self.trainer.cfg.TEST.METRIC_NAMES
        else:
            self.metric_name = tuple([self.trainer.cfg.DATASETS.TESTS[0] + "/{}".format(metric) for metric in tuple(self.trainer.cfg.TEST.METRIC_NAMES)])

    def after_epoch(self):
        # No way to use **kwargs
        storage = get_event_storage()
        metric_dict = dict()
        for metric in self.metric_name:
            metric_dict.update({
                metric: storage.latest()[metric][0] if metric in storage.latest() else -1
            })
        metric_dict.update({'epoch': self.trainer.epoch})
        self.step(self.trainer.iter - 1, **metric_dict)


class LRScheduler(HookBase):
    """
    A hook which executes a torch builtin LR scheduler and summarizes the LR.
    It is executed after every iteration.
    """

    def __init__(self, optimizer, scheduler):
        """
        Args:
            optimizer (torch.optim.Optimizer):
            scheduler (torch.optim._LRScheduler)
        """
        self._optimizer = optimizer
        self._scheduler = scheduler

        # NOTE: some heuristics on what LR to summarize
        # summarize the param group with most parameters
        largest_group = max(len(g["params"]) for g in optimizer.param_groups)

        if largest_group == 1:
            # If all groups have one parameter,
            # then find the most common initial LR, and use it for summary
            lr_count = Counter([g["lr"] for g in optimizer.param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer.param_groups):
                if g["lr"] == lr:
                    self._best_param_group_id = i
                    break
        else:
            for i, g in enumerate(optimizer.param_groups):
                if len(g["params"]) == largest_group:
                    self._best_param_group_id = i
                    break

    def after_step(self):
        lr = self._optimizer.param_groups[self._best_param_group_id]["lr"]
        self.trainer.storage.put_scalar("lr", lr, smoothing_hint=False)

    def after_epoch(self):
        next_epoch = self.trainer.epoch + 1
        if next_epoch <= self.trainer.warmup_epochs:
            self._scheduler["warmup_sched"].step()
        elif next_epoch >= self.trainer.delay_epochs:
            self._scheduler["lr_sched"].step()


class AutogradProfiler(HookBase):
    """
    A hook which runs `torch.autograd.profiler.profile`.
    Examples:
    .. code-block:: python
        hooks.AutogradProfiler(
             lambda trainer: trainer.iter > 10 and trainer.iter < 20, self.cfg.OUTPUT_DIR
        )
    The above example will run the profiler for iteration 10~20 and dump
    results to ``OUTPUT_DIR``. We did not profile the first few iterations
    because they are typically slower than the rest.
    The result files can be loaded in the ``chrome://tracing`` page in chrome browser.
    Note:
        When used together with NCCL on older version of GPUs,
        autograd profiler may cause deadlock because it unnecessarily allocates
        memory on every device it sees. The memory management calls, if
        interleaved with NCCL calls, lead to deadlock on GPUs that do not
        support `cudaLaunchCooperativeKernelMultiDevice`.
    """

    def __init__(self, enable_predicate, output_dir, *, use_cuda=True):
        """
        Args:
            enable_predicate (callable[trainer -> bool]): a function which takes a trainer,
                and returns whether to enable the profiler.
                It will be called once every step, and can be used to select which steps to profile.
            output_dir (str): the output directory to dump tracing files.
            use_cuda (bool): same as in `torch.autograd.profiler.profile`.
        """
        self._enable_predicate = enable_predicate
        self._use_cuda = use_cuda
        self._output_dir = output_dir

    def before_step(self):
        if self._enable_predicate(self.trainer):
            self._profiler = torch.autograd.profiler.profile(use_cuda=self._use_cuda)
            self._profiler.__enter__()
        else:
            self._profiler = None

    def after_step(self):
        if self._profiler is None:
            return
        self._profiler.__exit__(None, None, None)
        out_file = os.path.join(
            self._output_dir, "profiler-trace-iter{}.json".format(self.trainer.iter)
        )
        if "://" not in out_file:
            self._profiler.export_chrome_trace(out_file)
        else:
            # Support non-posix filesystems
            with tempfile.TemporaryDirectory(prefix="fastreid_profiler") as d:
                tmp_file = os.path.join(d, "tmp.json")
                self._profiler.export_chrome_trace(tmp_file)
                with open(tmp_file) as f:
                    content = f.read()
            with PathManager.open(out_file, "w") as f:
                f.write(content)


class EvalHook(HookBase):
    """
    Run an evaluation function periodically, and at the end of training.
    It is executed every ``eval_period`` iterations and after the last iteration.
    """

    def __init__(self, eval_period, eval_function, do_val=False, metric_names=('Rank-1', 'mAP')):
        """
        Args:
            eval_period (int): the period to run `eval_function`.
            eval_function (callable): a function which takes no arguments, and
                returns a nested dict of evaluation metrics.
        Note:
            This hook must be enabled in all or none workers.
            If you would like only certain workers to perform evaluation,
            give other workers a no-op function (`eval_function=lambda: None`).
        """
        self._period = eval_period
        self._func = eval_function
        self.logger = logging.getLogger(__name__)
        self.best_iteration = 0
        self.do_val = do_val
        self.metric_names = metric_names

    def _do_eval(self, val=False):
        if val:
            results = self._func(mode='val')
        else:
            results = self._func(mode='test')

        if results:
            assert isinstance(
                results, dict
            ), "Eval function must return a dict. Got {} instead.".format(results)

            flattened_results = flatten_results_dict(results)
            for k, v in flattened_results.items():
                try:
                    v = float(v)
                except Exception:
                    raise ValueError(
                        "[EvalHook] eval_function should return a nested dict of float. "
                        "Got '{}: {}' instead.".format(k, v)
                    )
            self.trainer.storage.put_scalars(**flattened_results, smoothing_hint=False)

        # Remove extra memory cache of main process due to evaluation
        torch.cuda.empty_cache()

    def after_epoch(self):
        next_epoch = self.trainer.epoch + 1
        is_final = next_epoch == self.trainer.max_epoch
        if is_final or (self._period > 0 and next_epoch % self._period == 0):
            self._do_eval(self.do_val)

            if comm.is_main_process():
                for metric_name in self.metric_names:
                    best_perf, self.best_iteration = max(self.trainer.storage.history(metric_name).values())
                    self.logger.info(f'Model on iteration {self.best_iteration} performs with {metric_name} as {best_perf:.4f}')
            # broadcast value from main process
            self.best_iteration = int(comm.broadcast_value(self.best_iteration, 0))

        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.
        comm.synchronize()

    def after_train(self):
        if self.do_val:
            self.logger.info(f"Run final testing with model in iteration {self.best_iteration}")
            # resume from the best model, and run the evaluation again
            if self.best_iteration + 1 == self.trainer.max_iter:
                weight_path = os.path.join(self.trainer.cfg.OUTPUT_DIR, "model_final.pth")
            else:
                weight_path = os.path.join(self.trainer.cfg.OUTPUT_DIR, "model_{:07d}.pth".format(self.best_iteration))
            if os.path.exists(weight_path):
                self.trainer.checkpointer.load(weight_path)
                self._do_eval()
            else:
                self.logger.warning("best model {} not exists.".format(weight_path))
        comm.synchronize()
        # func is likely a closure that holds reference to the trainer
        # therefore we clean it to avoid circular reference in the end
        del self._func


class PreciseBN(HookBase):
    """
    The standard implementation of BatchNorm uses EMA in inference, which is
    sometimes suboptimal.
    This class computes the true average of statistics rather than the moving average,
    and put true averages to every BN layer in the given model.
    It is executed after the last iteration.
    """

    def __init__(self, model, data_loader, num_iter):
        """
        Args:
            model (nn.Module): a module whose all BN layers in training mode will be
                updated by precise BN.
                Note that user is responsible for ensuring the BN layers to be
                updated are in training mode when this hook is triggered.
            data_loader (iterable): it will produce data to be run by `model(data)`.
            num_iter (int): number of iterations used to compute the precise
                statistics.
        """
        self._logger = logging.getLogger(__name__)
        if len(get_bn_modules(model)) == 0:
            self._logger.info(
                "PreciseBN is disabled because model does not contain BN layers in training mode."
            )
            self._disabled = True
            return

        self._model = model
        self._data_loader = data_loader
        self._num_iter = num_iter
        self._disabled = False

        self._data_iter = None

    def after_epoch(self):
        next_epoch = self.trainer.epoch + 1
        is_final = next_epoch == self.trainer.max_epoch
        if is_final:
            self.update_stats()

    def update_stats(self):
        """
        Update the model with precise statistics. Users can manually call this method.
        """
        if self._disabled:
            return

        if self._data_iter is None:
            self._data_iter = iter(self._data_loader)

        def data_loader():
            for num_iter in itertools.count(1):
                if num_iter % 100 == 0:
                    self._logger.info(
                        "Running precise-BN ... {}/{} iterations.".format(num_iter, self._num_iter)
                    )
                # This way we can reuse the same iterator
                yield next(self._data_iter)

        with EventStorage():  # capture events in a new storage to discard them
            self._logger.info(
                "Running precise-BN for {} iterations...  ".format(self._num_iter)
                + "Note that this could produce different statistics every time."
            )
            update_bn_stats(self._model, data_loader(), self._num_iter)


class FreezeLayer(HookBase):
    def __init__(self, model, open_layer_names, freeze_iters, fc_freeze_iters):
        self._logger = logging.getLogger(__name__)
        if hasattr(model, 'module'):
            model = model.module
        self.model = model

        self.open_layer_names = open_layer_names
        self.freeze_iters = freeze_iters
        self.fc_freeze_iters = fc_freeze_iters

        self.is_frozen = False
        self.fc_frozen = False
        # previous requires grad status
        param_grad = {}
        for name, param in self.model.named_parameters():
            param_grad[name] = param.requires_grad
        self.param_grad = param_grad

    def before_step(self):
        # Freeze specific layers
        if self.trainer.iter < self.freeze_iters and not self.is_frozen:
            self.freeze_specific_layer()

        # Recover original layers status
        if self.trainer.iter >= self.freeze_iters and self.is_frozen:
            self.open_all_layer()

        if self.trainer.max_iter - self.trainer.iter <= self.fc_freeze_iters \
                and not self.fc_frozen:
            self.freeze_classifier()
    def freeze_classifier(self):
        for p in self.model.heads.classifier.parameters():
            p.requires_grad_(False)

        self.fc_frozen = True
        self._logger.info("Freeze classifier training for "
                          "last {} iterations".format(self.fc_freeze_iters))
                          
    def freeze_specific_layer(self):
        open_layer_names = copy.deepcopy(self.open_layer_names)
        for name, module in self.model.named_modules():
            flag = False
            for layer in open_layer_names:
                if flag: continue
                if layer in name:
                    module.train()
                    for p in module.parameters():
                        p.requires_grad_(True)
                    flag = True
                else:
                    module.eval()
                    for p in module.parameters():
                        p.requires_grad_(False)

        self.is_frozen = True
        for layer in self.open_layer_names:
            for name, module in self.model.named_modules():
                if layer in name:
                    open_layer_names.remove(layer)
                    break
        if open_layer_names:
            self._logger.info(f'{open_layer_names} is not an attribute of the model, will skip this layer')
        # check by run
        self._logger.info('Print the parameters would be updated：')
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._logger.info(name)

    def open_all_layer(self):
        self.model.train()
        for name, param in self.model.named_parameters():
            param.requires_grad = self.param_grad[name]

        self.is_frozen = False

        self._logger.info(f'Open all layers for training')

class SWA(HookBase):
    def __init__(self, swa_start: int, swa_freq: int, swa_lr_factor: float, eta_min: float, lr_sched=False, ):
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_lr_factor = swa_lr_factor
        self.eta_min = eta_min
        self.lr_sched = lr_sched

    def before_step(self):
        is_swa = self.trainer.iter == self.swa_start
        if is_swa:
            # Wrapper optimizer with SWA
            self.trainer.optimizer = optim.SWA(self.trainer.optimizer, self.swa_freq, self.swa_lr_factor)
            self.trainer.optimizer.reset_lr_to_swa()

            if self.lr_sched:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer=self.trainer.optimizer,
                    T_0=self.swa_freq,
                    eta_min=self.eta_min,
                )

    def after_step(self):
        next_iter = self.trainer.iter + 1

        # Use Cyclic learning rate scheduler
        if next_iter > self.swa_start and self.lr_sched:
            self.scheduler.step()

        is_final = next_iter == self.trainer.max_iter
        if is_final:
            self.trainer.optimizer.swap_swa_param()


class LabelGeneratorHook(HookBase):
    """Generate pseudo labels 
    """
    __factory = {
        'dbscan': label_generator_dbscan,
        'kmeans': label_generator_kmeans,
        'cdp': label_generator_cdp,
        'hypergraph': label_generator_hypergraph_new
    }

    def __init__(self, cfg, model):
        self._logger = logging.getLogger(__name__)
        self._step_timer = Timer()
        self._cfg = cfg
        self.model = model
        # only build the data loader for unlabeled dataset
        self._logger.info("Build the dataloader for clustering.")
        self._data_loader_cluster = build_reid_train_loader(cfg, is_train=False, for_clustering=True)
        # save the original unlabeled dataset info
        self._common_dataset = self._data_loader_cluster.dataset

        assert cfg.PSEUDO.ENABLED, "pseudo label settings are not enabled."
        assert cfg.PSEUDO.NAME in self.__factory.keys(), \
            f"{cfg.PSEUDO.NAME} is not supported, please select from {self.__factory.keys()}"
        self.label_generator = self.__factory[cfg.PSEUDO.NAME]

        self.num_classes = []
        self.indep_thres = []
        if cfg.PSEUDO.NAME in ['kmeans', 'hypergraph']:
            self.num_classes = cfg.PSEUDO.NUM_CLUSTER

    def before_train(self):
        # save the original all dataset info
        self._logger.info("Copy the dataloader from trainer dataloader for building the new dataloader in training stage.")
        self._data_loader_sup = copy.deepcopy(self.trainer.data_loader)

    def before_epoch(self):
        if self.trainer.epoch % self._cfg.PSEUDO.CLUSTER_EPOCH == 0 \
                or self.trainer.epoch == self.trainer.start_epoch:
            self._step_timer.reset()

            # get memory features
            self.get_memory_features()

            # generate pseudo labels and centers
            all_labels, all_centers = self.update_labels()

            # update train loader
            self.update_train_loader(all_labels)

            # reset optimizer
            if self._cfg.PSEUDO.RESET_OPT:
                self._logger.info(f"Reset optimizer")
                self.trainer.optimizer.state = collections.defaultdict(dict)

            if hasattr(self.trainer, 'memory'):
                # update memory labels, memory based methods such as SpCL
                self.update_memory_labels(all_labels)
                assert len(all_centers) == 1, 'only support single unsupervised dataset'
                self.trainer.memory._update_center(all_centers[0])
            else:
                # update classifier centers, methods such as SBL
                self.update_classifier_centers(all_centers)

            comm.synchronize()

            sec = self._step_timer.seconds()
            self._logger.info(f"Finished updating pseudo label in {str(datetime.timedelta(seconds=int(sec)))}")

    def get_memory_features(self):
        # get memory features
        if hasattr(self.trainer, 'memory'):
            self.memory_features = []
            start_ind = 0
            # features for labeled dataset is not store in memory_features
            for idx, dataset in enumerate(self._data_loader_sup.dataset.datasets):
                if idx in self._cfg.PSEUDO.UNSUP:
                    self.memory_features.append(
                        self.trainer.memory.features[start_ind : start_ind + len(dataset)].clone().cpu()
                    )
                    start_ind += len(dataset)
                else:
                    start_ind += dataset.num_train_pids
            self.memory_features = torch.cat(self.memory_features)
            self.trainer.memory._update_epoch(self.trainer.epoch)
        else:
            self.memory_features = None

    def update_memory_labels(self, all_labels):
        sup_commdataset = self._data_loader_sup.dataset
        sup_datasets = sup_commdataset.datasets
        memory_labels = []
        start_pid = 0
        for idx, dataset in enumerate(sup_datasets):
            if idx in self._cfg.PSEUDO.UNSUP:
                labels = all_labels[self._cfg.PSEUDO.UNSUP.index(idx)]
                memory_labels.append(torch.LongTensor(labels) + start_pid)
                start_pid += max(labels) + 1
            else:
                num_pids = dataset.num_train_pids
                memory_labels.append(torch.arange(start_pid, start_pid + num_pids))
                start_pid += num_pids
        memory_labels = torch.cat(memory_labels).view(-1)
        self.trainer.memory._update_label(memory_labels)

    def update_labels(self):
        self._logger.info(f"Start updating pseudo labels on epoch {self.trainer.epoch}/iteration {self.trainer.iter}")
        
        if self.memory_features is None:
            all_features, true_labels, img_paths = extract_features(self.model,
                                                                    self._data_loader_cluster,
                                                                    self._cfg.PSEUDO.NORM_FEAT)
        else:
            all_features = self.memory_features
            true_labels = torch.LongTensor([item[1] for item in self._data_loader_cluster.dataset.img_items])

        if self._cfg.PSEUDO.NORM_FEAT:
            all_features = F.normalize(all_features, p=2, dim=1)
        datasets_size = self._common_dataset.datasets_size
        datasets_size_range = list(itertools.accumulate([0] + datasets_size))
        assert len(all_features) == datasets_size_range[-1], f"number of features {len(all_features)} should be same as the unlabeled data size {datasets_size_range[-1]}"

        all_centers = []
        all_labels = []
        all_dataset_names = [self._cfg.DATASETS.NAMES[ind] for ind in self._cfg.PSEUDO.UNSUP]
        for idx, dataset in enumerate(self._common_dataset.datasets):

            try:
                indep_thres = self.indep_thres[idx]
            except:
                indep_thres = None
            try:
                num_classes = self.num_classes[idx]
            except:
                num_classes = None

            if comm.is_main_process():
                # clustering only on first GPU
                save_path = '{}/clustering/{}/clustering_epoch{}.pt'.format(self._cfg.OUTPUT_DIR, all_dataset_names[idx], self.trainer.epoch)
                start_id, end_id = datasets_size_range[idx], datasets_size_range[idx + 1]
                # if os.path.exists(save_path):
                #     res = torch.load(save_path)
                #     labels, centers, num_classes, indep_thres, dist_mat = res['labels'], res['centers'], res['num_classes'], res['indep_thres'], res['dist_mat']
                # else:
                labels, centers, num_classes, indep_thres, dist_mat = self.label_generator(
                    self._cfg,
                    all_features[start_id: end_id],
                    num_classes=num_classes,
                    indep_thres=indep_thres,
                    epoch=self.trainer.epoch
                )
                if not os.path.exists(save_path):
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    res = {'labels': labels, 'num_classes': num_classes}#, 'centers': centers, 'indep_thres': indep_thres, 'dist_mat': dist_mat}
                    torch.save(res, save_path)

                if self._cfg.PSEUDO.NORM_CENTER:
                    centers = F.normalize(centers, p=2, dim=1)
            comm.synchronize()

            # broadcast to other process
            if comm.get_world_size() > 1:
                num_classes = int(comm.broadcast_value(num_classes, 0))
                if self._cfg.PSEUDO.NAME == "dbscan" and len(self._cfg.PSEUDO.DBSCAN.EPS) > 1:
                    # use clustering reliability criterion
                    indep_thres = comm.broadcast_value(indep_thres, 0)
                if comm.get_rank() > 0:
                    labels = torch.arange(len(dataset)).long()
                    centers = torch.zeros((num_classes, all_features.size(-1))).float()
                    if self._cfg.PSEUDO.MEMORY.WEIGHTED:
                        dist_mat = torch.zeros((len(dataset), len(dataset))).float()
                labels = comm.broadcast_tensor(labels, 0)
                centers = comm.broadcast_tensor(centers, 0)
                if self._cfg.PSEUDO.MEMORY.WEIGHTED:
                    dist_mat = comm.broadcast_tensor(dist_mat, 0)

            # calculate weight_matrix if use weight_matrix in loss calculation
            if self._cfg.PSEUDO.MEMORY.WEIGHTED:
                assert len(self._cfg.DATASETS.NAMES) == 1, 'Only single single dataset is supported for calculating weight_matrix'
                weight_matrix = torch.zeros((len(labels), num_classes))
                nums = torch.zeros((1, num_classes))
                index_select = torch.where(labels >= 0)[0]
                inputs_select = dist_mat[index_select]
                labels_select = labels[index_select]
                weight_matrix.index_add_(1, labels_select, inputs_select)
                nums.index_add_(1, labels_select, torch.ones(1, len(index_select)))
                weight_matrix = 1 - weight_matrix / nums
                self.trainer.weight_matrix = weight_matrix

            if comm.is_main_process():
                self.label_summary(labels, true_labels[start_id:end_id], indep_thres=indep_thres)
            all_labels.append(labels.tolist())
            all_centers.append(centers)

            try:
                self.indep_thres[idx] = indep_thres
            except:
                self.indep_thres.append(indep_thres)
            try:
                self.num_classes[idx] = num_classes
            except:
                self.num_classes.append(num_classes)

        return all_labels, all_centers

    def update_train_loader(self, all_labels):
        # Here is tricky, we take the datasets from self._data_loader_sup, this datasets is created same as supervised learning.
        sup_commdataset = self._data_loader_sup.dataset
        sup_datasets = sup_commdataset.datasets
        # add the ground truth labels into the all_labels
        pid_labels = list()
        start_pid = 0

        for idx, dataset in enumerate(sup_datasets):
            # get ground truth pid labels start from 0
            pid_lbls = [pid for _, pid, _ in dataset.data]
            min_pid = min(pid_lbls)
            pid_lbls = [pid_lbl - min_pid for pid_lbl in pid_lbls]
            
            if idx in self._cfg.PSEUDO.UNSUP:
                # replace gt labels with pseudo labels
                unsup_idx = self._cfg.PSEUDO.UNSUP.index(idx)
                pid_lbls = all_labels[unsup_idx]

            pid_lbls = [pid + start_pid if pid != -1 else pid for pid in pid_lbls]
            start_pid = max(pid_lbls) + 1
            pid_labels.append(pid_lbls)            

        self.trainer.data_loader = build_reid_train_loader(self._cfg,
                                                           datasets=copy.deepcopy(sup_datasets),  # copy the sup_datasets
                                                           pseudo_labels=pid_labels,
                                                           is_train=True,
                                                           relabel=False)
        self.trainer._data_loader_iter = iter(self.trainer.data_loader)
        # update cfg
        if self._cfg.is_frozen():
            self._cfg.defrost()
            self._cfg.MODEL.HEADS.NUM_CLASSES = self.trainer.data_loader.dataset.num_classes
            self._cfg.freeze()
        else:
            self._cfg.MODEL.HEADS.NUM_CLASSES = self.trainer.data_loader.dataset.num_classes

    def update_classifier_centers(self, all_centers):
        assert len(self._cfg.DATASETS.NAMES) == len(self._cfg.PSEUDO.UNSUP), 'Only support training on unlabeled datasets.'
        start_cls_id = 0
        for idx in range(len(self._cfg.DATASETS.NAMES)):
            if idx in self._cfg.PSEUDO.UNSUP:
                center_labels = torch.arange(start_cls_id, start_cls_id + self.trainer.data_loader.dataset.datasets[idx].num_train_pids)
                centers = all_centers[self._cfg.PSEUDO.UNSUP.index(idx)]
                if hasattr(self.model, 'module'):
                    self.model.module.initialize_centers(centers, center_labels)
                else:
                    self.model.initialize_centers(centers, center_labels)
                start_cls_id += self.trainer.data_loader.dataset.datasets[idx].num_train_pids

    def label_summary(self, pseudo_labels, gt_labels, cluster_metric=True, indep_thres=None):
        if cluster_metric:
            nmi_score, ari_score, purity_score, cluster_acc, precision, recall, fscore, precision_singular, recall_singular, fscore_singular = cluster_metrics(pseudo_labels.long().numpy(), gt_labels.long().numpy())
            self._logger.info(f"nmi_score: {nmi_score*100:.2f}%, ari_score: {ari_score*100:.2f}%, purity_score: {purity_score*100:.2f}%, cluster_acc: {cluster_acc*100:.2f}%, precision: {precision*100:.2f}%, recall: {recall*100:.2f}%, fscore: {fscore*100:.2f}%, precision_singular: {precision_singular*100:.2f}%, recall_singular: {recall_singular*100:.2f}%, fscore_singular: {fscore_singular*100:.2f}%.")
            self.trainer.storage.put_scalar('nmi_score', nmi_score, smoothing_hint=False)
            self.trainer.storage.put_scalar('ari_score', ari_score, smoothing_hint=False)
            self.trainer.storage.put_scalar('purity_score', purity_score, smoothing_hint=False)
            self.trainer.storage.put_scalar('cluster_acc', cluster_acc, smoothing_hint=False)
            self.trainer.storage.put_scalar('precision', precision, smoothing_hint=False)
            self.trainer.storage.put_scalar('recall', recall, smoothing_hint=False)
            self.trainer.storage.put_scalar('fscore', fscore, smoothing_hint=False)
            self.trainer.storage.put_scalar('precision_singular', precision_singular, smoothing_hint=False)
            self.trainer.storage.put_scalar('recall_singular', recall_singular, smoothing_hint=False)
            self.trainer.storage.put_scalar('fscore_singular', fscore_singular, smoothing_hint=False)

        # statistics of clusters and un-clustered instances
        index2label = collections.defaultdict(int)
        for label in pseudo_labels.tolist():
            index2label[label] += 1
        unused_ins_num = index2label.pop(-1) if -1 in index2label.keys() else 0
        index2label = np.array(list(index2label.values()))
        clu_num = (index2label > 1).sum()
        unclu_ins_num = (index2label == 1).sum()

        if indep_thres is None:
            self._logger.info(f"Cluster Statistic: clusters {clu_num}, un-clustered instances {unclu_ins_num}, unused instances {unused_ins_num}")
        else:
            self._logger.info(f"Cluster Statistic: clusters {clu_num}, un-clustered instances {unclu_ins_num}, unused instances {unused_ins_num}, R_indep threshold is {1 - indep_thres}")

        self.trainer.storage.put_scalar('num_clusters', clu_num, smoothing_hint=False)
        self.trainer.storage.put_scalar('num_outliers', unclu_ins_num, smoothing_hint=False)
        self.trainer.storage.put_scalar('num_unused_instances', unused_ins_num, smoothing_hint=False)

        return clu_num, unclu_ins_num, unused_ins_num