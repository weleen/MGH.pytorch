# encoding: utf-8
"""
@author:  tianjian
"""

import numpy as np
from .build import ACTIVE_SAMPLERS_REGISTRY


@ACTIVE_SAMPLERS_REGISTRY.register()
class RandomSampler:
    def __init__(self, cfg):
        self.triplet_set = set()
        self.K = cfg.ACTIVE.SAMPLE_K
        self.count = 0

    def sample(self, indexes, sim_mat, targets):
        res = []
        last_len = len(self.triplet_set)
        for i, index1 in enumerate(indexes):
            pos_set = []
            neg_set = []
            index1 = index1.item()
            mask = np.random.permutation(len(sim_mat[i]))
            for index2 in sim_mat[i][mask][:self.K]:
                index2 = index2.item()
                if targets[index2] == targets[index1]:
                    pos_set.append(index2)
                else:
                    neg_set.append(index2)
            res.append([len(pos_set), len(neg_set)])
            for pos in pos_set:
                for neg in neg_set:
                    self.triplet_set.add((index1, pos, neg))
        # total = 0
        # for line in res:
        #     total += (line[0] * line[1])
        # assert total == (len(self.triplet_set)-last_len)
        np.save('{}.npy'.format(self.count), res)
        self.count += 1