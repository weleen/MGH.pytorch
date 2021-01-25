import numpy as np
from .sampler import Sampler

class CoresetSampler(Sampler):
    def query_sample(self, dist_mat, pred_labels, pred_num_classes, targets, cluster_num=None, features=None, centers=None, temp=0.05) -> np.ndarray:
        ulbl_indexes = np.arange(self.data_size)[~self.index_label]
        lbl_indexes = np.arange(self.data_size)[self.index_label]
        m = ulbl_indexes.shape[0]
        
        if len(lbl_indexes) == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = dist_mat[ulbl_indexes, :][:, lbl_indexes].numpy()
            # dist_ctr = compute_distance_matrix(ulbl_features, lbl_features, metric='cosine').numpy()
            min_dist = np.amin(dist_ctr, axis=1)

        query_index = []
        for _ in range(self.query_num):
            idx = min_dist.argmax()
            query_index.append(idx)
            new_lbl_indexes = np.append(lbl_indexes, idx)
            dist_new_ctr = dist_mat[ulbl_indexes, :][:, new_lbl_indexes].numpy()
            min_dist = np.amin(dist_new_ctr, axis=1)

        return ulbl_indexes[query_index]