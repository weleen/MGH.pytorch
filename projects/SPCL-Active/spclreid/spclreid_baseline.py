# encoding: utf-8
"""
@author:  wuyiming
@contact: yimingwu@hotmail.com
"""

import torch
from torch import nn
from torch.nn import functional as F

from fastreid.layers import *
from fastreid.modeling.backbones import build_backbone
from fastreid.layers import get_norm
from fastreid.utils.torch_utils import weights_init_kaiming
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

        # fmt: off
        feat_dim      = cfg.MODEL.BACKBONE.FEAT_DIM
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        neck_feat     = cfg.MODEL.HEADS.NECK_FEAT
        pool_type     = cfg.MODEL.HEADS.POOL_LAYER
        with_bnneck   = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type     = cfg.MODEL.HEADS.NORM

        if pool_type == 'fastavgpool':   pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'avgpool':     pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'maxpool':     pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempoolP':    pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == 'gempool':     pool_layer = GeneralizedMeanPooling()
        elif pool_type == "avgmaxpool":  pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == 'clipavgpool': pool_layer = ClipGlobalAvgPool2d()
        elif pool_type == "identity":    pool_layer = nn.Identity()
        elif pool_type == "flatten":     pool_layer = Flatten()
        else:                            raise KeyError(f"{pool_type} is not supported!")
        # fmt: on

        self.heads = nn.Sequential()
        self.heads.add_module('pool_layer', pool_layer)
        self.neck_feat = neck_feat

        bottleneck = nn.Sequential()
        if embedding_dim > 0:
            bottleneck.add_module('conv1', nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim

        if with_bnneck:
            bottleneck.add_module('bnneck', get_norm(norm_type, feat_dim, bias_freeze=True))

        bottleneck.apply(weights_init_kaiming)
        self.heads.add_module('bottleneck', bottleneck)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)

        global_feat = self.heads.pool_layer(features)
        bn_feat = self.heads.bottleneck(global_feat)
        bn_feat = bn_feat[..., 0, 0]

        if self.neck_feat == "before":  feat = global_feat[..., 0, 0]
        elif self.neck_feat == "after": feat = bn_feat
        else:
            raise KeyError("MODEL.HEADS.NECK_FEAT value is invalid, must choose from ('after' & 'before')")

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"].long()

            # normalize the feature
            feat = F.normalize(feat)
            return feat, targets, batched_inputs['index']
        else:
            return bn_feat

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
