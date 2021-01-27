import torch
import numpy as np

def dist2classweight(dist_mat: torch.Tensor, num_classes: int, labels: torch.Tensor):
    """
    calculate class-wise weight matrix from dist matrix.
    """
    weight_matrix = torch.zeros((len(labels), num_classes))
    nums = torch.zeros((1, num_classes))
    if labels.min() < 0:
        index_select = torch.where(labels >= 0)[0]
        inputs_select = dist_mat[index_select]
        labels_select = labels[index_select]
        weight_matrix.index_add_(1, labels_select, inputs_select)
        nums.index_add_(1, labels_select, torch.ones(1, len(index_select)))
    else:
        weight_matrix.index_add_(1, labels, dist_mat)
        nums.index_add_(1, labels, torch.ones(1, len(labels)))
    weight_matrix = 1 -  weight_matrix / nums
    return weight_matrix

def set_labeled_instances(sampler, dist_mat: torch.Tensor, gt_labels: torch.Tensor):
    new_dist_mat = dist_mat
    labeled_dist_mat = -torch.ones_like(dist_mat)
    for i in sampler.query_set:
        for j in sampler.query_set:
            if gt_labels[i] == gt_labels[j]:
                new_dist_mat[i, j] = 0
                new_dist_mat[j, i] = 0
                labeled_dist_mat[i, j] = 0
                labeled_dist_mat[j, i] = 0
            else:
                new_dist_mat[i, j] = 1
                new_dist_mat[j, i] = 1
                labeled_dist_mat[i, j] = 1
                labeled_dist_mat[j, i] = 1
    return new_dist_mat, labeled_dist_mat

def set_labeled_instances_np(sampler, dist_mat: np.ndarray, gt_labels):
    new_dist_mat = dist_mat.copy()
    labeled_dist_mat = -np.ones_like(dist_mat)
    for i in sampler.query_set:
        for j in sampler.query_set:
            if gt_labels[i] == gt_labels[j]:
                new_dist_mat[i, j] = 0
                new_dist_mat[j, i] = 0
                labeled_dist_mat[i, j] = 0
                labeled_dist_mat[j, i] = 0
            else:
                new_dist_mat[i, j] = 1
                new_dist_mat[j, i] = 1
                labeled_dist_mat[i, j] = 1
                labeled_dist_mat[j, i] = 1
    return new_dist_mat, labeled_dist_mat