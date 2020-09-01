# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging
import torch
from torch import nn

from fastreid.layers import GeneralizedMeanPoolingP, AdaptiveAvgMaxPool2d, FastGlobalAvgPool2d
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_reid_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        # backbone
        self.backbone = build_backbone(cfg)

        # head
        pool_type = cfg.MODEL.HEADS.POOL_LAYER
        if pool_type == 'fastavgpool':  pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'avgpool':    pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'maxpool':    pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempool':    pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == "avgmaxpool": pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == "identity":   pool_layer = nn.Identity()
        else:
            raise KeyError(f"{pool_type} is invalid, please choose from "
                           f"'avgpool', 'maxpool', 'gempool', 'avgmaxpool' and 'identity'.")

        in_feat = cfg.MODEL.HEADS.IN_FEAT
        self.num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        self.heads = build_reid_heads(cfg, in_feat, self.num_classes, pool_layer)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"].long().to(self.device)

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            return self.heads(features, targets)
        else:
            return self.heads(features)

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs["images"].to(self.device)
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs.to(self.device)

        return images

    def losses(self, outputs, **kwargs):
        logits, feat, targets = outputs
        return reid_losses(self._cfg, logits, feat, targets)

    @torch.no_grad()
    def initialize_centers(self, centers, labels):
        logger = logging.getLogger(__name__)
        if self.num_classes > 0:
            self.heads.classifier.weight.data[labels.min().item(): labels.max().item() + 1].copy_(
                centers.to(self.heads.classifier.weight.device))
        else:
            logger.warning(
                f"there is no classifier in the {self.__class__.__name__}, "
                f"the initialization does not function"
            )
