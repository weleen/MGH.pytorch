'''
Author: WuYiming
Date: 2020-10-12 21:22:11
LastEditTime: 2020-10-14 16:06:27
LastEditors: Please set LastEditors
Description: Hooks for SpCL
FilePath: /fast-reid/projects/SpCL_new/hooks.py
'''

import datetime
import collections

import torch

from fastreid.utils import comm
from fastreid.engine.hooks import *

from fastreid.engine.train_loop import HookBase
from fastreid.data.samplers import InferenceSampler
from fastreid.data.build import fast_batch_collator
from fastreid.evaluation.evaluator import inference_context
from samplers import build_active_samplers
import random


class SpCLLabelGeneratorHook(LabelGeneratorHook):

    def before_step(self):
        if self.trainer.epoch % self._cfg.PSEUDO.CLUSTER_EPOCH == 0 \
                or self.trainer.iter == self.trainer.start_iter:
            self._step_timer.reset()

            # get memory features
            self.get_memory_features()

            # generate pseudo labels and centers
            all_labels, all_centers = self.update_labels()

            # update train loader
            self.update_train_loader(all_labels)

            # update memory labels
            self.update_memory_labels(all_labels)

            comm.synchronize()

            sec = self._step_timer.seconds()
            self._logger.info(f"Finished updating pseudo label in {str(datetime.timedelta(seconds=int(sec)))}")

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


class ActiveHook(HookBase):

    def __init__(self, cfg, data_len):
        super().__init__()
        self.base_iter = cfg.ACTIVE.TRAIN_ITER * cfg.DATALOADER.ITERS_PER_EPOCH
        index_set = list(range(data_len))
        random.shuffle(index_set)
        index_set = index_set[:int(data_len * cfg.ACTIVE.SAMPLE_M)]
        self.index_set = index_set
        self.samplers = build_active_samplers(cfg, index_set)
        labeled_num = int(len(index_set) * cfg.ACTIVE.INITIAL_RATE) + 1
        index_dataloader = torch.utils.data.DataLoader(index_set, batch_size=labeled_num, shuffle=True)
        self._index_iter = iter(index_dataloader)
        self.warmup_iters = cfg.ACTIVE.WARMUP_ITER * cfg.DATALOADER.ITERS_PER_EPOCH
        self.active_max_iters = self.base_iter / cfg.ACTIVE.INITIAL_RATE + self.warmup_iters

    def before_step(self):
        if self.warmup_iters <= self.trainer.iter < self.active_max_iters and self.trainer.iter % self.base_iter == 0:
            indexes = self._index_iter.next()
            all_features, targets = self.get_feature()
            features = all_features[self.index_set]
            sel_feats = all_features[indexes]
            dist_mat = self.euclidean_dist(sel_feats, features)
            # only choose first 30 similar instances
            sim_mat = torch.argsort(dist_mat, dim=1)[:, 1:31]
            self.samplers.sample(indexes, sim_mat, targets)
            self.trainer.data_loader_active = self.trainer.build_active_sample_dataloader(self.samplers.triplet_set, is_train=True)
            self.trainer._data_loader_active_iter = iter(self.trainer.data_loader_active)
            self.trainer.active_warmup = True
            
    def get_feature(self):
        num_workers = self.trainer.cfg.DATALOADER.NUM_WORKERS
        batch_size = self.trainer.cfg.TEST.IMS_PER_BATCH
        data_sampler = InferenceSampler(len(self.trainer.data_loader.dataset))
        batch_sampler = torch.utils.data.BatchSampler(data_sampler, batch_size, False)
        dataloader = torch.utils.data.DataLoader(
            self.trainer.data_loader.dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=fast_batch_collator,
        )

        features = []
        targets = []
        model = self.trainer.model
        with inference_context(model), torch.no_grad():
            for idx, inputs in enumerate(dataloader):
                outputs = model(inputs)
                features.append(outputs)
                targets.append(inputs['targets'])
            features = torch.cat(features)
            targets = torch.cat(targets)
        return features, targets
    
    def euclidean_dist(self, x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(x, y.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist