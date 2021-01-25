import numpy as np
import torch
from .sampler import Sampler
from .utils import dist2classweight

class EntropySampler(Sampler):
    def query_sample(self, dist_mat, pred_labels, pred_num_classes, targets, cluster_num=None, features=None, centers=None, temp=0.05) -> np.ndarray:
        sim_mat = dist2classweight(dist_mat, pred_num_classes, pred_labels) / temp
        if cluster_num:
            sim_mat = sim_mat[:, :cluster_num]

        ulbl_indexes = np.arange(self.data_size)[~self.index_label]
        probs = sim_mat[ulbl_indexes]
        log_probs = torch.log(probs)
        U = (probs*log_probs).sum(1)
        query_index = ulbl_indexes[U.sort()[1][:self.query_num]]
        return query_index