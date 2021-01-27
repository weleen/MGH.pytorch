# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from torch import nn

from fastreid.utils import comm


class CircleLoss(object):
    def __init__(self, cfg):
        self._scale = cfg.MODEL.LOSSES.CIRCLE.SCALE

        self._m = cfg.MODEL.LOSSES.CIRCLE.MARGIN
        self._s = cfg.MODEL.LOSSES.CIRCLE.ALPHA

    def __call__(self, _, embedding, targets):
        embedding = nn.functional.normalize(embedding, dim=1)

        if comm.get_world_size() > 1:
            all_embedding = comm.concat_all_gather(embedding)
            all_targets = comm.concat_all_gather(targets)
        else:
            all_embedding = embedding
            all_targets = targets

        dist_mat = torch.matmul(all_embedding, all_embedding.t())

        N = dist_mat.size(0)
        is_pos = all_targets.view(N, 1).expand(N, N).eq(all_targets.view(N, 1).expand(N, N).t()).float()

        # Compute the mask which ignores the relevance score of the query to itself
        is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

        is_neg = all_targets.view(N, 1).expand(N, N).ne(all_targets.view(N, 1).expand(N, N).t())

        s_p = dist_mat * is_pos
        s_n = dist_mat * is_neg

        alpha_p = torch.clamp_min(-s_p.detach() + 1 + self._m, min=0.)
        alpha_n = torch.clamp_min(s_n.detach() + self._m, min=0.)
        delta_p = 1 - self._m
        delta_n = self._m

        logit_p = - self._s * alpha_p * (s_p - delta_p)
        logit_n = self._s * alpha_n * (s_n - delta_n)

        loss = nn.functional.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

        return {
            "loss_circle": loss * self._scale,
        }


def pairwise_circleloss(
        embedding: torch.Tensor,
        targets: torch.Tensor,
        margin: float,
        gamma: float, ) -> torch.Tensor:
    embedding = F.normalize(embedding, dim=1)

    dist_mat = torch.matmul(embedding, embedding.t())

    N = dist_mat.size(0)

    is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
    is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

    # Mask scores related to itself
    is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

    s_p = dist_mat * is_pos
    s_n = dist_mat * is_neg

    alpha_p = torch.clamp_min(-s_p.detach() + 1 + margin, min=0.)
    alpha_n = torch.clamp_min(s_n.detach() + margin, min=0.)
    delta_p = 1 - margin
    delta_n = margin

    logit_p = - gamma * alpha_p * (s_p - delta_p) + (-99999999.) * (1 - is_pos)
    logit_n = gamma * alpha_n * (s_n - delta_n) + (-99999999.) * (1 - is_neg)

    loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

    return loss


def pairwise_cosface(
        embedding: torch.Tensor,
        targets: torch.Tensor,
        margin: float,
        gamma: float, ) -> torch.Tensor:
    # Normalize embedding features
    embedding = F.normalize(embedding, dim=1)

    dist_mat = torch.matmul(embedding, embedding.t())

    N = dist_mat.size(0)
    is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
    is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

    # Mask scores related to itself
    is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

    s_p = dist_mat * is_pos
    s_n = dist_mat * is_neg

    logit_p = -gamma * s_p + (-99999999.) * (1 - is_pos)
    logit_n = gamma * (s_n + margin) + (-99999999.) * (1 - is_neg)

    loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

    return loss
    