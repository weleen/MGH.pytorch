import numpy as np
from .sampler import Sampler

class RandomSampler(Sampler):
    def query_sample(self, dist_mat, pred_labels, pred_num_classes, targets, cluster_num=None, features=None, centers=None, temp=0.05) -> np.ndarray:
        indexes = np.where(self.index_label==0)[0]
        query_index = indexes[np.random.permutation(len(indexes))][:self.query_num]
        return query_index