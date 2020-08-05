# encoding: utf-8
"""
@author:  tianjian
"""

from torch import nn
from torch.nn import init
from torch.nn import functional as F

from fastreid.layers import *
from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier
from fastreid.modeling.heads.build import REID_HEADS_REGISTRY


@REID_HEADS_REGISTRY.register()
class BUCHead(nn.Module):
    def __init__(self, cfg, in_feat, num_classes, pool_layer=nn.AdaptiveAvgPool2d(1)):
        super().__init__()
        self.neck_feat = cfg.MODEL.HEADS.NECK_FEAT
        self.pool_layer = pool_layer
        bottle_dim = 2048
        self.bottle_neck = nn.Linear(in_feat, bottle_dim)
        self.bnneck = nn.BatchNorm1d(bottle_dim)
        init.kaiming_normal_(self.bottle_neck.weight, mode='fan_out')
        init.constant_(self.bottle_neck.bias, 0)
        init.constant_(self.bnneck.weight, 1)
        init.constant_(self.bnneck.bias, 0)
        self.drop = nn.Dropout()

        # identity classification layer
        cls_type = cfg.MODEL.HEADS.CLS_LAYER
        if cls_type == 'linear':    self.classifier = nn.Linear(bottle_dim, num_classes, bias=False)
        elif cls_type == 'arcface': self.classifier = Arcface(cfg, bottle_dim, num_classes)
        elif cls_type == 'circle':  self.classifier = Circle(cfg, bottle_dim, num_classes)
        else:
            raise KeyError(f"{cls_type} is invalid, please choose from "
                           f"'linear', 'arcface' and 'circle'.")

        self.classifier.apply(weights_init_classifier)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        global_feat = self.pool_layer(features)
        global_feat = global_feat[..., 0, 0]
        eval_feat = F.normalize(global_feat, p=2, dim=1)
        global_feat = self.bottle_neck(global_feat)
        bn_feat = self.bnneck(global_feat)
        bn_feat = F.normalize(bn_feat, p=2, dim=1)
        bn_feat = self.drop(bn_feat)
        # Evaluation
        if not self.training: return eval_feat
        # Training
        try:              pred_class_logits = self.classifier(bn_feat)
        except TypeError: pred_class_logits = self.classifier(bn_feat, targets)

        if self.neck_feat == "before":  feat = global_feat
        elif self.neck_feat == "after": feat = bn_feat
        else:
            raise KeyError("MODEL.HEADS.NECK_FEAT value is invalid, must choose from ('after' & 'before')")
        return pred_class_logits, feat, targets
