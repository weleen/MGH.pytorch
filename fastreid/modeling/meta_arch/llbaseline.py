# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging
import torch
from torch import nn

from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from fastreid.layers import *
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class LLBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        self.num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        # backbone
        resnet = build_backbone(cfg)
        self.backbone1 = nn.Sequential(
          resnet.conv1,
          resnet.bn1,
          resnet.relu,
          resnet.maxpool,
          resnet.layer1,
        )
        self.backbone2 = nn.Sequential(
          resnet.layer2,
          resnet.layer3,
          resnet.layer4
        )

        pool_type = cfg.MODEL.HEADS.POOL_LAYER
        # low-level feature
        if pool_type == 'fastavgpool':   self.low_level_pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'avgpool':     self.low_level_pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'maxpool':     self.low_level_pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempoolP':    self.low_level_pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == 'gempool':     self.low_level_pool_layer = GeneralizedMeanPooling()
        elif pool_type == "avgmaxpool":  self.low_level_pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == 'clipavgpool': self.low_level_pool_layer = ClipGlobalAvgPool2d()
        elif pool_type == "identity":    self.low_level_pool_layer = nn.Identity()
        elif pool_type == "flatten":     self.low_level_pool_layer = Flatten()
        else:                            raise KeyError(f"{pool_type} is not supported!")

        # head
        self.heads = build_heads(cfg)

    @property
    def device(self):
        return self.backbone1[0].weight.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        x = self.backbone1(images)
        features = self.backbone2(x)
        low_level_feature = self.low_level_pool_layer(x)[..., 0, 0]

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"].long().to(self.device)

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(features, targets)
            return {
                "outputs": outputs,
                "targets": targets,
                "low_level_feature": low_level_feature
            }
        else:
            outputs = self.heads(features)
            return outputs

    def preprocess_image(self, batched_inputs):
        r"""
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs["images"].to(self.device)
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs.to(self.device)
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        return images

    def losses(self, outs, **kwargs):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        return reid_losses(self._cfg, outs, **kwargs)

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
