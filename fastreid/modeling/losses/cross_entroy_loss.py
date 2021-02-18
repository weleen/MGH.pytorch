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
        self._tau = cfg.MODEL.LOSSES.CE.TAU
        self._start_epoch = cfg.MODEL.LOSSES.CE.START_EPOCH

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

        log_probs = F.log_softmax(pred_class_logits / self._tau, dim=1)
        with torch.no_grad():
            targets = torch.ones_like(log_probs)
            targets *= smooth_param / (num_classes - 1)
            targets.scatter_(1, gt_classes.data.unsqueeze(1), (1 - smooth_param))

        loss = (-targets * log_probs).sum(dim=1)

        with torch.no_grad():
            non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)

        loss = loss.sum() / non_zero_cnt

        if 'epoch' in kwargs and kwargs['epoch'] >= self._start_epoch:
            return {
                "loss_cls": loss * self._scale,
            }
        else:
            return {
                "loss_cls": torch.tensor([0]).to(pred_class_logits.device)
            }


class SoftEntropyLoss(object):
    """
    A class that stores information and compute losses about outputs of a head.
    """
    def __init__(self, cfg):
        super(SoftEntropyLoss, self).__init__()
        self._scale = cfg.MODEL.LOSSES.CE.SCALE
        self._tau = cfg.MODEL.LOSSES.SCE.TAU
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.softmax = torch.nn.Softmax(dim=1)

    def __call__(self, pred_class_logits, _, targets, **kwargs):
        assert 'outs_mean' in kwargs, 'outs_mean not found in input, only {}'.format(kwargs.keys())
        targets = kwargs['outs_mean']['outputs']['cls_outputs']
        log_probs = self.logsoftmax(pred_class_logits / self._tau)
        assert targets.size(1) == log_probs.size(1)
        loss = (-self.softmax(targets / self._tau).detach() * log_probs).mean(0).sum()
        if 'epoch' in kwargs and kwargs['epoch'] >= self._start_epoch:
            return {
                "loss_soft_cls": loss * self._scale
            }
        else:
            return {
                "loss_soft_cls": torch.tensor([0]).to(pred_class_logits.device)
            }


class HardViewContrastiveLoss(object):
    """
    Reference: ICE: Inter-instance Contrastive Encoding for Unsupervised Person Re-identification, CVPR2021 submission
    """
    def __init__(self, cfg):
        self._scale = cfg.MODEL.LOSSES.VCL.SCALE
        self._tau = cfg.MODEL.LOSSES.VCL.TAU
        self._normalized_features = cfg.MODEL.LOSSES.VCL.NORM_FEAT
        self._start_epoch = cfg.MODEL.LOSSES.VCL.START_EPOCH

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
        embedding_sim = torch.exp(embedding.mm(all_embedding_mean.t()) / self._tau)
        N, M = embedding_sim.size()

        is_pos = targets.view(N, 1).expand(N, M).eq(all_targets.view(M, 1).expand(M, N).t()).float()
        masked_pos = (embedding_sim * is_pos + 9999.0 * (1 - is_pos)).min(dim=1)[0]
        masked_neg = (embedding_sim * (1 - is_pos)).sum(dim=1)

        if 'epoch' in kwargs and kwargs['epoch'] >= self._start_epoch:
            return {
                'loss_vcl': self._scale * -torch.log(masked_pos / (masked_neg + masked_pos)).mean()
            }
        else:
            return {
                'loss_vcl': torch.tensor([0]).to(embedding.device)
            }
