# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""
from numpy.lib.arraysetops import unique
import torch
import torch.nn.functional as F

from fastreid.utils.events import get_event_storage
from fastreid.utils import comm


class CrossEntropyLoss(object):
    """
    A class that stores information and compute losses about outputs of a Baseline head.
    """

    def __init__(self, cfg):
        self._eps = cfg.MODEL.LOSSES.CE.EPSILON
        self._alpha = cfg.MODEL.LOSSES.CE.ALPHA
        self._scale = cfg.MODEL.LOSSES.CE.SCALE

    @staticmethod
    def log_accuracy(pred_class_logits, gt_classes, topk=(1,), name='cls_accuracy'):
        """
        Log the accuracy metrics to EventStorage.
        """
        bsz = pred_class_logits.size(0)
        maxk = max(topk)
        _, pred_class = pred_class_logits.topk(maxk, 1, True, True)
        pred_class = pred_class.t()
        correct = pred_class.eq(gt_classes.view(1, -1).expand_as(pred_class))

        ret = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
            ret.append(correct_k.mul_(1. / bsz))

        storage = get_event_storage()
        storage.put_scalar(name, ret[0])

    def __call__(self, pred_class_logits, _, gt_classes, **kwargs):
        """
        Compute the softmax cross entropy loss.
        Returns:
            scalar Tensor
        """
        num_classes = pred_class_logits.size(1)
        self.log_accuracy(pred_class_logits, gt_classes)
        if self._eps >= 0:
            smooth_param = self._eps
        else:
            # Adaptive label smooth regularization
            soft_label = F.softmax(pred_class_logits, dim=1)
            smooth_param = self._alpha * soft_label[torch.arange(soft_label.size(0)), gt_classes].unsqueeze(1)

        log_probs = F.log_softmax(pred_class_logits, dim=1)
        with torch.no_grad():
            targets = torch.ones_like(log_probs)
            targets *= smooth_param / (num_classes - 1)
            targets.scatter_(1, gt_classes.data.unsqueeze(1), (1 - smooth_param))

        loss = (-targets * log_probs).sum(dim=1)

        with torch.no_grad():
            non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)

        loss = loss.sum() / non_zero_cnt

        return {
            "loss_cls": loss * self._scale,
        }


class SoftEntropyLoss(object):
    """
    A class that stores information and compute losses about outputs of a head.
    """
    def __init__(self, cfg):
        super(SoftEntropyLoss, self).__init__()
        self._scale = cfg.MODEL.LOSSES.CE.SCALE
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.softmax = torch.nn.Softmax(dim=1)

    def __call__(self, pred_class_logits, _, targets, **kwargs):
        assert 'outs_mean' in kwargs, 'outs_mean not found in input, only {}'.format(kwargs.keys())
        targets = kwargs['outs_mean']['outputs']['cls_outputs']
        log_probs = self.logsoftmax(pred_class_logits)
        assert targets.size(1) == log_probs.size(1)
        loss = (-self.softmax(targets).detach() * log_probs).mean(0).sum()
        return {
            "loss_soft_cls": loss * self._scale
        }


class CenterContrastiveLoss(object):
    """
    Reference: ICE: Inter-instance Contrastive Encoding for Unsupervised Person Re-identification, CVPR2021 submission
    """
    def __init__(self, cfg):
        self.scale = cfg.MODEL.LOSSES.CCL.SCALE
        self.tau = cfg.MODEL.LOSSES.CCL.TAU
        self._normalized_features = cfg.MODEL.LOSSES.CCL.NORM_FEAT

    def __call__(self, _, embedding, targets, eps=1e-6, **kwargs):
        assert 'outs_mean' in kwargs, 'outs_mean not found in input, only {}'.format(kwargs.keys())
        results_mean = kwargs['outs_mean']
        embedding_mean = results_mean['outputs']['features']
        if self._normalized_features:
            embedding = F.normalize(embedding, dim=1)
            embedding_mean = F.normalize(embedding_mean, dim=1)

        # For distributed training, gather all features from different process.
        if comm.get_world_size() > 1:
            all_targets = comm.concat_all_gather(targets)
            all_embedding_mean = comm.concat_all_gather(embedding_mean)
        else:
            all_targets = targets
            all_embedding_mean = embedding_mean

        # def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
        #     exps = torch.exp(vec)
        #     masked_exps = exps * mask.float().clone()
        #     masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
        #     return masked_exps / masked_sums

        # embedding_sim = embedding.mm(all_embedding_mean.t()) / self.tau
        # # simple version
        # unique_targets = all_targets.unique()
        # mapping_targets = {}
        # for index, c in enumerate(unique_targets.tolist()):
        #     mapping_targets[c] = index
        # new_targets = torch.Tensor([mapping_targets[c] for c in targets.tolist()]).long().cuda()
        # new_all_targets = torch.Tensor([mapping_targets[c] for c in all_targets.tolist()]).long().cuda()

        # sim = torch.zeros(embedding.size(0), new_all_targets.max() + 1).float().cuda()
        # sim.index_add_(1, new_all_targets, embedding_sim)
        # nums = torch.zeros(1, new_all_targets.max() + 1).float().cuda()
        # nums.index_add_(1, new_all_targets, torch.ones(1, new_all_targets.size(0)).float().cuda())

        # mask = (nums > 0).float()
        # sim /= (mask * nums + (1 - mask)).clone().expand_as(sim)
        # mask = mask.expand_as(sim)
        # masked_sim = masked_softmax(sim, mask)

        # direct
        unique_targets = all_targets.unique().tolist()
        centers = []
        for i in unique_targets:
            indexes = torch.where(all_targets == i)[0]
            centers.append(all_embedding_mean[indexes].mean(0))
        centers = torch.stack(centers).contiguous()
        if self._normalized_features:
            centers = F.normalize(centers, dim=1)
        centers_simi = torch.exp(embedding.mm(centers.t()) / self.tau)
        center_index = torch.Tensor([unique_targets.index(t) for t in targets]).view(-1, 1).long().cuda()
        numerator = torch.gather(centers_simi, 1, center_index)
        loss = -torch.log(numerator / centers_simi.sum(dim=1)).mean()

        return {
            'loss_ccl': self.scale * loss,
            # 'loss_ccl': self.scale * F.nll_loss(torch.log(masked_sim + eps), new_targets)
        }


class HardViewContrastiveLoss(object):
    """
    Reference: ICE: Inter-instance Contrastive Encoding for Unsupervised Person Re-identification, CVPR2021 submission
    """
    def __init__(self, cfg):
        self.scale = cfg.MODEL.LOSSES.VCL.SCALE
        self.tau = cfg.MODEL.LOSSES.VCL.TAU
        self._normalized_features = cfg.MODEL.LOSSES.VCL.NORM_FEAT

    def __call__(self, _, embedding, targets, eps=1e-6, **kwargs):
        assert 'outs_mean' in kwargs, 'outs_mean not found in input, only {}'.format(kwargs.keys())
        results_mean = kwargs['outs_mean']
        embedding_mean = results_mean['outputs']['features']
        if self._normalized_features:
            embedding = F.normalize(embedding, dim=1)
            embedding_mean = F.normalize(embedding_mean, dim=1)

        # For distributed training, gather all features from different process.
        if comm.get_world_size() > 1:
            all_targets = comm.concat_all_gather(targets)
            all_embedding_mean = comm.concat_all_gather(embedding_mean)
        else:
            all_targets = targets
            all_embedding_mean = embedding_mean
        embedding_sim = torch.exp(embedding.mm(all_embedding_mean.t()) / self.tau)

        unique_targets = all_targets.unique()
        mapping_targets = {}
        for index, c in enumerate(unique_targets.tolist()):
            mapping_targets[c] = index
        new_targets = torch.Tensor([mapping_targets[c] for c in targets.tolist()]).long().cuda()
        new_all_targets = torch.Tensor([mapping_targets[c] for c in all_targets.tolist()]).long().cuda()

        mask = (new_targets.view(-1, 1) == new_all_targets.view(1, -1)).float()
        masked_pos = (embedding_sim * mask + 99999 * (1 - mask)).min(dim=1)[0]
        masked_neg = (embedding_sim * (1 - mask)).sum(dim=1)

        return {
            'loss_vcl': self.scale * -torch.log(masked_pos / (masked_neg + masked_pos)).mean()
        }