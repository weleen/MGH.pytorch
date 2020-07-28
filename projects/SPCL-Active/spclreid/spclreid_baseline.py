# encoding: utf-8
"""
@author:  wuyiming
@contact: yimingwu@hotmail.com
"""

import torch
from torch import nn
from torch.nn import functional as F

from fastreid.layers import GeneralizedMeanPoolingP, AdaptiveAvgMaxPool2d, FastGlobalAvgPool2d
from fastreid.modeling.backbones import build_backbone
from fastreid.layers import get_norm
from fastreid.utils.weight_init import weights_init_kaiming
from fastreid.modeling.meta_arch.build import META_ARCH_REGISTRY
from .active_triplet_loss import ActiveTripletLoss


@META_ARCH_REGISTRY.register()
class USL_Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))
        self._cfg = cfg
        # backbone
        self.backbone = build_backbone(cfg)

        # head
        pool_type = cfg.MODEL.HEADS.POOL_LAYER
        if pool_type == 'avgpool':      pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'maxpool':    pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempool':    pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == "avgmaxpool": pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == "identity":   pool_layer = nn.Identity()
        else:
            raise KeyError(f"{pool_type} is invalid, please choose from "
                           f"'avgpool', 'maxpool', 'gempool', 'avgmaxpool' and 'identity'.")

        self.pool_layer = pool_layer
        in_feat = cfg.MODEL.HEADS.IN_FEAT
        self.bnneck = get_norm(cfg.MODEL.HEADS.NORM, in_feat, cfg.MODEL.HEADS.NORM_SPLIT, bias_freeze=True)
        self.bnneck.apply(weights_init_kaiming)
        self.neck_feat = cfg.MODEL.HEADS.NECK_FEAT

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
            # try:              return pred_feat, batched_inputs["targets"], batched_inputs["camid"]
            # except Exception: return pred_feat

        images = self.preprocess_image(batched_inputs)
        targets = batched_inputs["targets"].long()

        # training
        features = self.backbone(images)  # (bs, 2048, 16, 8)
        global_feat = self.pool_layer(features)
        bn_feat = self.bnneck(global_feat)
        bn_feat = bn_feat[..., 0, 0]

        if self.neck_feat == "before":  feat = global_feat[..., 0, 0]
        elif self.neck_feat == "after": feat = bn_feat
        else:
            raise KeyError("MODEL.HEADS.NECK_FEAT value is invalid, must choose from ('after' & 'before')")
        # normalize the feature
        feat = F.normalize(feat)

        return feat, targets, batched_inputs['index']

    def inference(self, batched_inputs):
        assert not self.training
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)  # (bs, 2048, 16, 8)
        global_feat = self.pool_layer(features)
        bn_feat = self.bnneck(global_feat)
        pred_feat = bn_feat[..., 0, 0]
        return pred_feat

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        images = batched_inputs["images"].to(self.device)
        return images

    def losses_u(self, outputs, memory):
        feat, targets, indexes = outputs
        loss = memory(feat, indexes)
        return {'contrastive_loss': loss}

    def losses_a(self, outputs):
        feat, targets, indexes = outputs
        loss = ActiveTripletLoss(self._cfg)(indexes, feat, targets)
        return {'active_tri_loss': loss}
