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
from fastreid.utils.torch_utils import weights_init_kaiming
from fastreid.modeling.meta_arch.build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class USL_Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()       # backbone
        self.backbone = build_backbone(cfg)

        # head
        feat_dim      = cfg.MODEL.BACKBONE.FEAT_DIM
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        neck_feat     = cfg.MODEL.HEADS.NECK_FEAT
        pool_type     = cfg.MODEL.HEADS.POOL_LAYER
        with_bnneck   = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type     = cfg.MODEL.HEADS.NORM

        if pool_type == 'fastavgpool':  pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'avgpool':    pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'maxpool':    pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempool':    pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == "avgmaxpool": pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == "identity":   pool_layer = nn.Identity()
        else:
            raise KeyError(f"{pool_type} is invalid, please choose from "
                           f"'avgpool', 'maxpool', 'gempool', 'avgmaxpool' and 'identity'.")

        self.heads = nn.Sequential()
        self.heads.add_module('pool_layer', pool_layer)
        self.neck_feat = neck_feat

        bottleneck = []
        if embedding_dim > 0:
            bottleneck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim

        if with_bnneck:
            bottleneck.append(get_norm(norm_type, feat_dim, bias_freeze=True))

        bnneck = nn.Sequential(*bottleneck)
        bnneck.apply(weights_init_kaiming)
        self.heads.add_module('bnneck', bnneck)

    @property
    def device(self):
        return self.backbone.conv1.weight.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)

        global_feat = self.heads.pool_layer(features)
        bn_feat = self.heads.bnneck(global_feat)
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
        if isinstance(batched_inputs, dict):
            images = batched_inputs["images"]
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        return images

    def losses(self, outputs, memory):
        feat, targets, indexes = outputs
        loss = memory(feat, indexes)
        return {'contrastive_loss': loss}
