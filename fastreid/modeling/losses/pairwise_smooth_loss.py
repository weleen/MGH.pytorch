"""
@author: wuyiming
@contact: yimingwu@hotmail.com
"""
import torch
import torch.nn.functional as F


class PairwiseSmoothLoss(object):
    """
    Refer to the paper ``Weakly Supervised Person Re-identification: Cost-effective Learning with A New Benchmark, TNNLS2020.''
    https://github.com/wanggrun/SYSU-30k/blob/master/GraphReID/UnaryTerm_PairwiseTerm_WeaklyTripletLoss/loss/CrossEntropyLoss.py
    """
    def __init__(self, cfg):
        self._scale = cfg.MODEL.LOSSES.PS.SCALE
        self._sigma = cfg.MODEL.LOSSES.PS.SIGMA

    def __call__(self, pred_class_logits, embedding, gt_classes):
        """
        embedding is the low level features, such as output of resnet.layer1.
        """
        n = pred_class_logits.size(0)

        input_softmax = F.softmax(pred_class_logits, dim=1)
        # dist
        dist = torch.pow(embedding, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(embedding, embedding.t(), beta=1, alpha=-2)
        dist = torch.exp(- dist / self._sigma)

        # inner_product
        # inner_prod = -torch.matmul(input_softmax, torch.log(input_softmax.t()))
        inner_prod = -torch.matmul(input_softmax, input_softmax.t())

        # mask
        if len(gt_classes.shape) == 1:
            # index target
            gt_classes = F.one_hot(gt_classes, pred_class_logits.size(1))
            gt_classes = torch.argmax(gt_classes * input_softmax, -1)
            mask = gt_classes.expand(n, n).eq(gt_classes.expand(n, n).t())
        elif len(gt_classes.shape) == 2:
            # one-hot or soft label
            mask = torch.mm(gt_classes, gt_classes.t())
            mask /= (mask.max() + 1e-6)  # normalize
        else:
            raise ValueError('gt_classes.shape is {}, which is not supported.'.format(gt_classes.shape))
        mask = 1.0 - mask.float()

        loss = dist * inner_prod * mask
        loss = loss.mean()
        return {
            "loss_pairwise_smooth": loss * self._scale
        }