# encoding: utf-8
"""
@author:  tianjian
"""
import itertools
from typing import Optional

import numpy as np
from torch.utils.data import Sampler


class ActiveTripletSampler(Sampler):
    def __init__(self, pair_set: set, shuffle: bool = True, seed: Optional[int] = None):
        self.pair_set = list(pair_set)
        self._size = len(pair_set)
        assert self._size > 0
        self._shuffle = shuffle
        if seed is None:
            seed = np.random.randint(2 ** 31)
        self._seed = int(seed)

    def __iter__(self):
        for i in itertools.islice(self._infinite_indices(), 0, None, 1):
            yield from self.pair_set[i]

    def _infinite_indices(self):
        np.random.seed(self._seed)
        while True:
            if self._shuffle:
                yield from np.random.permutation(self._size)
            else:
                yield from np.arange(self._size)
