# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import random
from collections import defaultdict
import copy

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class BalancedIdentitySampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances=4):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances

        self.index_pid = defaultdict(list)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)

        for index, info in enumerate(data_source):
            pid = info[1]
            camid = info[2]
            self.index_pid[index] = pid
            self.pid_cam[pid].append(camid)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_identities = len(self.pids)

        self._seed = 0
        self._shuffle = True

    def __iter__(self):
        indices = self._infinite_indices()
        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])
            _, i_pid, i_cam = self.data_source[i]
            ret = [i]
            pid_i = self.index_pid[i]
            cams = self.pid_cam[pid_i]
            index = self.pid_index[pid_i]
            select_cams = No_index(cams, i_cam)

            if select_cams:
                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=True)
                for kk in cam_indexes:
                    ret.append(index[kk])
            else:
                select_indexes = No_index(index, i)
                if not select_indexes:
                    # only one image for this identity
                    ind_indexes = [0] * (self.num_instances - 1)
                elif len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=True)

                for kk in ind_indexes:
                    ret.append(index[kk])
            yield from ret

    def _infinite_indices(self):
        np.random.seed(self._seed)
        while True:
            if self._shuffle:
                identities = np.random.permutation(self.num_identities)
            else:
                identities = np.arange(self.num_identities)
            drop_indices = self.num_identities % self.num_pids_per_batch
            if drop_indices == 0:
                yield from identities
            yield from identities[:-drop_indices]


class NaiveIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances

        self.index_pid = defaultdict(list)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)

        for index, info in enumerate(data_source):
            pid = info[1]
            camid = info[2]
            self.index_pid[index] = pid
            self.pid_cam[pid].append(camid)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_identities = len(self.pids)

        self._seed = 0

    def __iter__(self):
        np.random.seed(self._seed)

        while True:
            batch_idxs_dict = defaultdict(list)

            for pid in self.pids:
                idxs = copy.deepcopy(self.pid_index[pid])
                if len(idxs) < self.num_instances:
                    idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                random.shuffle(idxs)
                batch_idxs = []
                for idx in idxs:
                    batch_idxs.append(idx)
                    if len(batch_idxs) == self.num_instances:
                        batch_idxs_dict[pid].append(batch_idxs)
                        batch_idxs = []

            avai_pids = copy.deepcopy(self.pids)
            final_idxs = []

            while len(avai_pids) >= self.num_pids_per_batch:
                selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
                for pid in selected_pids:
                    batch_idxs = batch_idxs_dict[pid].pop(0)
                    final_idxs.extend(batch_idxs)
                    if len(batch_idxs_dict[pid]) == 0:
                        avai_pids.remove(pid)
            yield from final_idxs


class RandomMultipleGallerySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances

        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)

        for index, (_, pid, cam) in enumerate(data_source):
            if (pid < 0): continue
            self.index_pid[index] = pid
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_identities = len(self.pids)
        self._seed = 0

    def __iter__(self):
        np.random.seed(self._seed)

        while True:
            indices = torch.randperm(self.num_identities).tolist()
            ret = []

            for kid in indices:
                i = random.choice(self.pid_index[self.pids[kid]])
                _, i_pid, i_cam = self.data_source[i]
                ret.append(i)

                pid_i = self.index_pid[i]
                cams = self.pid_cam[pid_i]
                index = self.pid_index[pid_i]
                select_cams = No_index(cams, i_cam)

                if select_cams:
                    if len(select_cams) >= self.num_instances:
                        cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=False)
                    else:
                        cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=True)

                    for kk in cam_indexes:
                        ret.append(index[kk])
                else:
                    select_indexes = No_index(index, i)
                    if (not select_indexes): continue
                    if len(select_indexes) >= self.num_instances:
                        ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=False)
                    else:
                        ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=True)

                    for kk in ind_indexes:
                        ret.append(index[kk])

            yield from ret
