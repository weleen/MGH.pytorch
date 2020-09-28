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
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        self.num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        # backbone
        self.backbone = build_backbone(cfg)

        # head
        self.heads = build_heads(cfg)

    @property
    def device(self):
        return self.backbone.conv1.weight.device

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

            outputs = self.heads(features, targets)
            return {
                "outputs": outputs,
                "targets": targets,
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

    def losses(self, outs):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # fmt: off
        outputs           = outs["outputs"]
        gt_labels         = outs["targets"]
        # model predictions
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs       = outputs['cls_outputs']
        pred_features     = outputs['features']
        # fmt: on
        return reid_losses(self._cfg, cls_outputs, pred_features, gt_labels)

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
