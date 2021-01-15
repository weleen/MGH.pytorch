# encoding: utf-8
'''
Author: WuYiming
Date: 2020-10-28 00:21:29
LastEditTime: 2020-11-21 01:11:39
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /fast-reid/projects/ConcurrentReID/concurrentreid/modeling/concurrentreid_head.py
'''
import torch
import torch.nn.functional as F
from torch import nn

from fastreid.layers import *
from fastreid.utils.torch_utils import weights_init_kaiming, weights_init_classifier
from fastreid.modeling.heads.build import REID_HEADS_REGISTRY


@REID_HEADS_REGISTRY.register()
class ConcurrentHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # fmt: off
        feat_dim      = cfg.MODEL.BACKBONE.FEAT_DIM
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        num_classes   = cfg.MODEL.HEADS.NUM_CLASSES if not cfg.PSEUDO.ENABLED or (cfg.PSEUDO.ENABLED and cfg.PSEUDO.WITH_CLASSIFIER) else 0
        neck_feat     = cfg.MODEL.HEADS.NECK_FEAT
        pool_type     = cfg.MODEL.HEADS.POOL_LAYER
        cls_type      = cfg.MODEL.HEADS.CLS_LAYER
        with_bnneck   = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type     = cfg.MODEL.HEADS.NORM

        if pool_type == 'fastavgpool':   self.pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'avgpool':     self.pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'maxpool':     self.pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempoolP':    self.pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == 'gempool':     self.pool_layer = GeneralizedMeanPooling()
        elif pool_type == "avgmaxpool":  self.pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == 'clipavgpool': self.pool_layer = ClipGlobalAvgPool2d()
        elif pool_type == "identity":    self.pool_layer = nn.Identity()
        elif pool_type == "flatten":     self.pool_layer = Flatten()
        else:                            raise KeyError(f"{pool_type} is not supported!")
        # fmt: on

        self.neck_feat = neck_feat

        self.bottleneck = nn.Sequential()
        if embedding_dim > 0:
            self.bottleneck.add_module('conv1', nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim

        if with_bnneck:
            self.bottleneck.add_module('bnneck', get_norm(norm_type, feat_dim, bias_freeze=True))
        self.bottleneck.apply(weights_init_kaiming)

        # identity classification layer
        if num_classes > 0:
            # fmt: off
            if cls_type == 'linear':          self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
            elif cls_type == 'arcSoftmax':    self.classifier = ArcSoftmax(cfg, feat_dim, num_classes)
            elif cls_type == 'circleSoftmax': self.classifier = CircleSoftmax(cfg, feat_dim, num_classes)
            elif cls_type == 'amSoftmax':     self.classifier = AMSoftmax(cfg, feat_dim, num_classes)
            else:                             raise KeyError(f"{cls_type} is not supported!")
            # fmt: on
            self.classifier.apply(weights_init_classifier)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        if self.cfg.CONCURRENT.ENABLED and self.training:
            bs_h, bs_w = self.cfg.CONCURRENT.BLOCK_SIZE
            b, c, hh, ww = features.size()
            h, w = hh // bs_h, ww // bs_w
            features = features.view(b, c, bs_h, h, bs_w, w)
            features = features.transpose(3, 4).reshape(b, c, bs_h * bs_w, h, w).transpose(1, 2).reshape(b * bs_h * bs_w, c, h, w).contiguous()
        global_feat = self.pool_layer(features)
        bn_feat = self.bottleneck(global_feat)
        bn_feat = bn_feat[..., 0, 0]

        # Evaluation
        # fmt: off
        if not self.training: return bn_feat
        # fmt: on

        # fmt: off
        if self.neck_feat == "before":  feat = global_feat[..., 0, 0]
        elif self.neck_feat == "after": feat = bn_feat
        else:                           raise KeyError(f"{self.neck_feat} is invalid for MODEL.HEADS.NECK_FEAT")
        # fmt: on

        outputs = {"features": feat}
        # Training
        if hasattr(self, "classifier"):
            if self.classifier.__class__.__name__ == 'Linear':
                cls_outputs = self.classifier(bn_feat)
                pred_class_logits = F.linear(bn_feat, self.classifier.weight)
            else:
                cls_outputs = self.classifier(bn_feat, targets)
                pred_class_logits = self.classifier.s * F.linear(F.normalize(bn_feat),
                                                                F.normalize(self.classifier.weight))
            outputs.update(
                {
                    "cls_outputs": cls_outputs,
                    "pred_class_logits": pred_class_logits,
                }
            )
        return outputs