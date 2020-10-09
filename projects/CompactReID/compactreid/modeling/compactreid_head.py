# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from torch import nn

from fastreid.layers import *
from fastreid.utils.torch_utils import weights_init_kaiming, weights_init_classifier
from fastreid.modeling.heads.build import REID_HEADS_REGISTRY


@REID_HEADS_REGISTRY.register()
class CompactHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # fmt: off
        feat_dim      = cfg.MODEL.BACKBONE.FEAT_DIM
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        num_classes   = cfg.MODEL.HEADS.NUM_CLASSES
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

        # identity classification layer
        # fmt: off
        if cls_type == 'linear':          self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        elif cls_type == 'arcSoftmax':    self.classifier = ArcSoftmax(cfg, feat_dim, num_classes)
        elif cls_type == 'circleSoftmax': self.classifier = CircleSoftmax(cfg, feat_dim, num_classes)
        elif cls_type == 'amSoftmax':     self.classifier = AMSoftmax(cfg, feat_dim, num_classes)
        else:                             raise KeyError(f"{cls_type} is not supported!")
        # fmt: on

        # compact classifier layer
        self.compact_classifier = nn.Sequential()
        self.compact_classifier.add_module('linear1', nn.Linear(feat_dim, feat_dim // 4, bias=True))
        self.compact_classifier.add_module('dropout1', nn.Dropout(0.5))
        self.compact_classifier.add_module('linear2', nn.Linear(feat_dim // 4, 2, bias=True))

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.compact_classifier.apply(weights_init_classifier)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        global_feat = self.pool_layer(features)
        bn_feat = self.bottleneck(global_feat)
        bn_feat = bn_feat[..., 0, 0]

        # Evaluation
        # fmt: off
        if not self.training: return bn_feat
        # fmt: on

        # Training
        if self.classifier.__class__.__name__ == 'Linear':
            cls_outputs = self.classifier(bn_feat)
            pred_class_logits = F.linear(bn_feat, self.classifier.weight)
        else:
            cls_outputs = self.classifier(bn_feat, targets)
            pred_class_logits = self.classifier.s * F.linear(F.normalize(bn_feat),
                                                             F.normalize(self.classifier.weight))
        compact_cls_outputs = self.compact_classifier(global_feat[..., 0, 0])
        # compact_cls_outputs = self.compact_classifier(bn_feat)
        # fmt: off
        if self.neck_feat == "before":  feat = global_feat[..., 0, 0]
        elif self.neck_feat == "after": feat = bn_feat
        else:                           raise KeyError(f"{self.neck_feat} is invalid for MODEL.HEADS.NECK_FEAT")
        # fmt: on

        b = features.size(0) // (self.cfg.COMPACT.TIMES + 1)
        cls_outputs = cls_outputs.view(b, -1, *cls_outputs.size()[1:])
        pred_class_logits = pred_class_logits.view(b, -1, *pred_class_logits.size()[1:])
        feat = feat.view(b, -1, *feat.size()[1:])

        # 0 means original image feature, 1 means jigsaw image feature
        compact_labels = torch.zeros((b, (self.cfg.COMPACT.TIMES + 1)), dtype=features.dtype).to(features.device)
        compact_labels[:, 1:] += 1
        compact_labels = compact_labels.view(-1)
        compact_labels_list = [compact_labels]
        for i in range(self.cfg.COMPACT.TIMES):
            compact_labels_list.append(1 - compact_labels)
        compact_labels = torch.stack(compact_labels_list, dim=1).contiguous()
        return {
            "cls_outputs": cls_outputs,
            "pred_class_logits": pred_class_logits,
            "features": feat,
            "compact_cls_outputs": compact_cls_outputs,
            "compact_labels": compact_labels
        }
