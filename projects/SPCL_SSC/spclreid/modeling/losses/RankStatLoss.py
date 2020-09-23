import torch
import torch.nn.functional as F


class RankStatLoss(torch.nn.Module):
    """
    Implementation of AUTOMATICALLY DISCOVERING AND LEARNING NEW VISUAL CATEGORIES WITH RANKING STATISTICS
    from https://github.com/k-han/AutoNovel
    """
    def __init__(self, cfg):
        super(RankStatLoss, self).__init__()
        self.criterion = torch.nn.BCELoss()
        self.topk = cfg.UNSUPERVISED.TOPK

    def pair_enum(self, x):
        """
        expand input tensor with the first dimension size
        :param x: (N, D)
        :return:
        x1: (N, N, D)
        x2: (N, N, D)
        """
        x1 = x.repeat(x.size(0), 1).view(-1, x.size(1))
        x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
        return x1, x2

    def forward(self, feat1, feat2, prob1, prob2):

        # get rank statistic
        rank_feat = feat1.detach()
        rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
        rank_idx1, rank_idx2 = self.pair_enum(rank_idx)
        rank_idx1 = rank_idx1[:, :self.topk]
        rank_idx2 = rank_idx2[:, :self.topk]
        rank_idx1, _ = torch.sort(rank_idx1, dim=1)
        rank_idx2, _ = torch.sort(rank_idx2, dim=1)

        rank_diff = torch.sum(torch.abs(rank_idx1 - rank_idx2), dim=1)
        target = torch.ones_like(rank_diff).float().to(feat1.device)
        target[rank_diff > 0] = 0

        prob1_expand, _ = self.pair_enum(prob1)
        _, prob2_expand = self.pair_enum(prob2)

        pred_sim = (prob1_expand * prob2_expand).sum(1)
        return {'rankstatistic_loss': self.criterion(pred_sim, target)}