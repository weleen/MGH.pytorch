# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F

from fastreid.utils import comm, euclidean_dist


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


def hard_example_mining(dist_mat, is_pos, return_indices=False, bound=9999.0):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pair wise distance between samples, shape [N, M]
      is_pos: positive index with shape [N, M]
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    """
    assert dist_mat.abs().max() < bound, 'dist_mat should be bounded in [-{}, {}]'.format(bound, bound)
    sorted_dist_mat, positive_indices = torch.sort(
        dist_mat + -bound * (1 - is_pos), dim=1, descending=True
    )
    dist_ap = sorted_dist_mat[:, 0]
    relative_p_inds = positive_indices[:, 0]
    sorted_dist_mat, negative_indices = torch.sort(
        dist_mat + bound * (is_pos), dim=1, descending=False
    )
    dist_an = sorted_dist_mat[:, 0]
    relative_n_inds = negative_indices[:, 0]
    if return_indices:
        return dist_ap, dist_an, relative_p_inds, relative_n_inds
    return dist_ap, dist_an


def weighted_example_mining(dist_mat, is_pos, weight_mining=False):
    """For each anchor, find the weighted positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, M]
      is_pos: positive index with shape [N, M]
      weight_mining: enable weighted mining or not
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    """
    assert len(dist_mat.size()) == 2

    is_pos = is_pos
    is_neg = 1 - is_pos
    dist_ap = dist_mat * is_pos
    dist_an = dist_mat * is_neg

    weights_ap = softmax_weights(dist_ap, is_pos)
    weights_an = softmax_weights(-dist_an, is_neg)

    if weight_mining:
        dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
        dist_an = torch.sum(dist_an * weights_an, dim=1)
    else:
        dist_ap = torch.sum(dist_ap, dim=1)
        dist_an = torch.sum(dist_an, dim=1)

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""
    def __init__(self, cfg):
        self._margin = cfg.MODEL.LOSSES.TRI.MARGIN
        self._normalize_feature = cfg.MODEL.LOSSES.TRI.NORM_FEAT
        self._scale = cfg.MODEL.LOSSES.TRI.SCALE
        self._hard_mining = cfg.MODEL.LOSSES.TRI.HARD_MINING
        self._weight_mining = cfg.MODEL.LOSSES.TRI.WEIGHT_MINING

    def __call__(self, _, embedding, targets, **kwargs):
        if self._normalize_feature: embedding = F.normalize(embedding, dim=1)

        # For distributed training, gather all features from different process.
        if comm.get_world_size() > 1:
            all_embedding = comm.concat_all_gather(embedding)
            all_targets = comm.concat_all_gather(targets)
        else:
            all_embedding = embedding
            all_targets = targets

        dist_mat = euclidean_dist(embedding, all_embedding)

        N, N = dist_mat.size()
        is_pos = targets.view(N, 1).expand(N, N).eq(all_targets.view(N, 1).expand(N, N).t())

        if self._hard_mining:
            dist_ap, dist_an = hard_example_mining(dist_mat, is_pos)
        else:
            dist_ap, dist_an = weighted_example_mining(dist_mat, is_pos, self._weight_mining)

        y = dist_an.new().resize_as_(dist_an).fill_(1)

        if self._margin > 0:
            loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=self._margin)
        else:
            loss = F.soft_margin_loss(dist_an - dist_ap, y)
            if loss == float('Inf'): loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)

        return {
            "loss_triplet": loss * self._scale,
        }


class SoftmaxTripletLoss(object):
    def __init__(self, cfg):
        self._margin = cfg.MODEL.LOSSES.STRI.MARGIN
        self._normalize_feature = cfg.MODEL.LOSSES.STRI.NORM_FEAT
        self._scale = cfg.MODEL.LOSSES.STRI.SCALE
        self._tau = cfg.MODEL.LOSSES.STRI.TAU

    def __call__(self, _, embedding, targets, **kwargs):
        assert 'outs_mean' in kwargs, 'outs_mean not found in input, only {}'.format(kwargs.keys())
        if self._normalize_feature:
            embedding = F.normalize(embedding, dim=1)

        # For distributed training, gather all features from different process.
        if comm.get_world_size() > 1:
            all_embedding = comm.concat_all_gather(embedding)
            all_targets = comm.concat_all_gather(targets)
        else:
            all_embedding = embedding
            all_targets = targets
        dist_mat = euclidean_dist(embedding, all_embedding)

        N, M = dist_mat.size()
        is_pos = targets.view(N, 1).expand(N, M).eq(all_targets.view(M, 1).expand(M, N).t()).float()

        dist_ap, dist_an = hard_example_mining(dist_mat, is_pos)

        triplet_dist = F.log_softmax(torch.stack((dist_ap, dist_an), dim=1) / self._tau, dim=1)
        loss = (-self._margin * triplet_dist[:, 0] - (1 - self._margin) * triplet_dist[:, 1]).mean()
        return {
            "loss_softmax_triplet": loss * self._scale,
        }


class SoftSoftmaxTripletLoss(object):
    def __init__(self, cfg):
        self._margin = cfg.MODEL.LOSSES.SSTRI.MARGIN
        self._normalize_feature = cfg.MODEL.LOSSES.SSTRI.NORM_FEAT
        self._scale = cfg.MODEL.LOSSES.SSTRI.SCALE
        self._tau = cfg.MODEL.LOSSES.SSTRI.TAU

    def __call__(self, _, embedding, targets, **kwargs):
        assert 'outs_mean' in kwargs, 'outs_mean not found in input, only {}'.format(kwargs.keys())
        results_mean = kwargs['outs_mean']
        embedding_mean = results_mean['outputs']['features']
        if self._normalize_feature:
            embedding = F.normalize(embedding, dim=1)
            embedding_mean = F.normalize(embedding_mean, dim=1)

        # For distributed training, gather all features from different process.
        if comm.get_world_size() > 1:
            all_embedding = comm.concat_all_gather(embedding)
            all_targets = comm.concat_all_gather(targets)
            all_embedding_mean = comm.concat_all_gather(embedding_mean)
        else:
            all_embedding = embedding
            all_targets = targets
            all_embedding_mean = embedding_mean

        dist_mat = euclidean_dist(embedding, all_embedding)

        N, M = dist_mat.size()
        is_pos = targets.view(N, 1).expand(N, M).eq(all_targets.view(M, 1).expand(M, N).t()).float()

        dist_ap, dist_an, ap_idx, an_idx = hard_example_mining(dist_mat, is_pos, return_indices=True)
        triplet_dist = F.log_softmax(torch.stack((dist_ap, dist_an), dim=1) / self._tau, dim=1)

        # reference from mean_net
        dist_mat_ref = euclidean_dist(embedding_mean, all_embedding_mean)
        dist_ap_ref = torch.gather(dist_mat_ref, 1, ap_idx.view(N, 1).expand(N, M))[:, 0]
        dist_an_ref = torch.gather(dist_mat_ref, 1, an_idx.view(N, 1).expand(N, M))[:, 0]

        triplet_dist_ref = F.softmax(torch.stack((dist_ap_ref, dist_an_ref), dim=1) / self._tau, dim=1).detach()
        loss = (-triplet_dist_ref * triplet_dist).mean(0).sum()
        return {
            "loss_soft_softmax_triplet": loss * self._scale,
        }


class ActiveTripletLoss(object):
    def __init__(self, cfg):
        # TODO: add in default config
        self._margin = 0.3
        self._scale = 1.0

    def __call__(self, _, global_features, targets):
        _, dim = global_features.size()
        global_features = global_features.view(-1, 3, dim)

        anchors = global_features[:, 0]
        positive = global_features[:, 1]
        negative = global_features[:, 2]

        loss = F.triplet_margin_loss(anchors, positive, negative, margin=self._margin)

        return {
            "loss_triplet_a": loss * self._scale,
        }