import numpy as np
import torch
import collections
from fastreid.utils.metrics import compute_distance_matrix
from .sampler import Sampler

class ClusterSampler(Sampler):
    def query_sample(self, dist_mat, pred_labels, pred_num_classes, targets, cluster_num=None, features=None, centers=None, temp=0.05) -> np.ndarray:
        img_per_cluster = 1
        center_labels = list(collections.Counter(pred_labels.tolist()).items())
        center_labels.sort(key=lambda d:d[1], reverse=True)
        
        sel_clusters = [d[0] for d in center_labels[:cluster_num]]
        
        dist = compute_distance_matrix(centers[sel_clusters], features)
        indexes = dist.sort(dim=1)[1]

        labeled_idx = []
        for i, cluster in enumerate(sel_clusters):
            labeled_idx.append(indexes[i][:img_per_cluster])
        labeled_idx = torch.cat(labeled_idx)
        return labeled_idx
