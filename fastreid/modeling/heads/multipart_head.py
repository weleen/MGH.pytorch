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
from .build import REID_HEADS_REGISTRY


@REID_HEADS_REGISTRY.register()
class MultiPartHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # fmt: off
        feat_dim      = cfg.MODEL.BACKBONE.FEAT_DIM
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        num_classes   = cfg.MODEL.HEADS.NUM_CLASSES if not cfg.PSEUDO.ENABLED or (cfg.PSEUDO.ENABLED and cfg.PSEUDO.WITH_CLASSIFIER) else 0
        neck_feat     = cfg.MODEL.HEADS.NECK_FEAT
        pool_type     = cfg.MODEL.HEADS.POOL_LAYER
        cls_type      = cfg.MODEL.HEADS.CLS_LAYER
        with_bnneck   = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type     = cfg.MODEL.HEADS.NORM
        self.num_part = cfg.MODEL.HEADS.NUM_PART

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
            elif cls_type == 'cosSoftmax':    self.classifier = CosSoftmax(cfg, feat_dim, num_classes)
            else:                             raise KeyError(f"{cls_type} is not supported!")
            # fmt: on
            self.classifier.apply(weights_init_classifier)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        global_feats = []
        bn_feats = []
        for num in self.num_part:
            assert features.size(2) % num == 0
            features_split = torch.split(features, features.size(2) // num, dim=2)
            local_feats = [self.pool_layer(f) for f in features_split]
            global_feats.extend([l[..., 0, 0] for l in local_feats])
            bn_feats.extend([self.bottleneck(f)[..., 0, 0] for f in local_feats])

        # Evaluation
        if not self.training: return bn_feats

        if self.neck_feat == "before":  feat = global_feats
        elif self.neck_feat == "after": feat = bn_feats
        else:                           raise KeyError(f"{self.neck_feat} is invalid for MODEL.HEADS.NECK_FEAT")

        outputs = {"features": feat,
                   "cls_outputs": None,
                   "pred_class_logits": None,
                   "global_features": global_feats}
        # Training
        if hasattr(self, "classifier"):
            if self.classifier.__class__.__name__ == 'Linear':
                cls_outputs = [self.classifier(bn_feat) for bn_feat in bn_feats]
                pred_class_logits = [F.linear(bn_feat, self.classifier.weight) for bn_feat in bn_feats]
            else:
                cls_outputs = [self.classifier(bn_feat, targets) for bn_feat in bn_feats]
                pred_class_logits = [self.classifier.s * F.linear(F.normalize(bn_feat),
                                                                F.normalize(self.classifier.weight)) for bn_feat in bn_feats]
            outputs.update(
                {
                    "cls_outputs": cls_outputs,
                    "pred_class_logits": pred_class_logits
                }
            )
        return outputs
