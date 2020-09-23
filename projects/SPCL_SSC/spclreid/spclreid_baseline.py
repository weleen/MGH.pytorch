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
from fastreid.utils.misc import rampup
from .modeling.losses import RankStatLoss


@META_ARCH_REGISTRY.register()
class USL_Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()       # backbone
        self.backbone = build_backbone(cfg)
        self._cfg = cfg
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

        num_cluster = cfg.UNSUPERVISED.CLUSTER_SIZE
        classifier = torch.nn.Linear(feat_dim, num_cluster)
        classifier.apply(weights_init_kaiming)
        self.heads.add_module('classifier', classifier)

    @property
    def device(self):
        return self.backbone.conv1.weight.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        images_size = images.size()
        if len(images_size) > 4:
            images = images.view(-1, *images_size[2:])
        features = self.backbone(images)

        global_feat = self.heads.pool_layer(features)
        bn_feat = self.heads.bottleneck(global_feat)
        bn_feat = bn_feat[..., 0, 0]
        pred_logits = self.heads.classifier(bn_feat)

        if self.neck_feat == "before":  feat = global_feat[..., 0, 0]
        elif self.neck_feat == "after": feat = bn_feat
        else:
            raise KeyError("MODEL.HEADS.NECK_FEAT value is invalid, must choose from ('after' & 'before')")

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"].long()

            # normalize the feature
            feat = F.normalize(feat)

            if len(images_size) > 4:
                feat = feat.view(images_size[0], images_size[1], *feat.size()[1:])
                pred_logits = pred_logits.view(images_size[0], images_size[1], *pred_logits.size()[1:])
                return feat, targets, batched_inputs["index"], pred_logits
            else:
                return feat, targets, batched_inputs["index"], pred_logits
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
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))
        return images

    def losses(self, outputs, memory, iters):
        feat, targets, indexes, pred_logits = outputs
        feat1 = feat[:, 0].contiguous()
        feat2 = feat[:, 1].contiguous()
        loss_dict = dict()
        loss_dict.update({'contrastrive_loss': memory(feat1, indexes)})
        prob1 = F.softmax(pred_logits[:, 0].contiguous(), dim=1)
        prob2 = F.softmax(pred_logits[:, 1].contiguous(), dim=1)
        rampup_scale = rampup(iters, self._cfg.UNSUPERVISED.RAMPUP_ITER, self._cfg.UNSUPERVISED.RAMPUP_COEFF)
        loss_dict.update({'consistency_loss': F.mse_loss(prob1, prob2) * rampup_scale})
        loss_dict.update(RankStatLoss(self._cfg)(feat1, feat2, prob1, prob2))
        return loss_dict