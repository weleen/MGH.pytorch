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
from fastreid.utils.misc import rampup
from fastreid.modeling.meta_arch.build import META_ARCH_REGISTRY

from .modeling.losses.NTXentLoss import NTXentLoss
from .modeling.losses import RankStatLoss


@META_ARCH_REGISTRY.register()
class SSCBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        # backbone
        self.backbone = build_backbone(cfg)

        # head
        pool_type = cfg.MODEL.HEADS.POOL_LAYER
        if pool_type == 'avgpool':
            pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'maxpool':
            pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempool':
            pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == "avgmaxpool":
            pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == "identity":
            pool_layer = nn.Identity()
        else:
            raise KeyError(f"{pool_type} is invalid, please choose from "
                           f"'avgpool', 'maxpool', 'gempool', 'avgmaxpool' and 'identity'.")

        self.pool_layer = pool_layer
        in_feat = cfg.MODEL.HEADS.IN_FEAT
        self.bnneck = get_norm(cfg.MODEL.HEADS.NORM, in_feat, cfg.MODEL.HEADS.NORM_SPLIT, bias_freeze=True)
        self.bnneck.apply(weights_init_kaiming)
        self.neck_feat = cfg.MODEL.HEADS.NECK_FEAT

        num_cluster = cfg.UNSUPERVISED.CLUSTER_SIZE
        self.classifier = torch.nn.Linear(in_feat, num_cluster)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        images_size = images.size()
        if len(images_size) > 4:
            images = images.view(-1, *images_size[2:])
        features = self.backbone(images)

        global_feat = self.pool_layer(features)
        bn_feat = self.bnneck(global_feat)
        bn_feat = bn_feat[..., 0, 0]
        pred_logits = self.classifier(bn_feat)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"].long().to(self.device)

            if self.neck_feat == "before":
                feat = global_feat[..., 0, 0]
            elif self.neck_feat == "after":
                feat = bn_feat
            else:
                raise KeyError("MODEL.HEADS.NECK_FEAT value is invalid, must choose from ('after' & 'before')")

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
        images = batched_inputs["images"].to(self.device)
        return images

    def losses(self, outputs, iter):
        feat, targets, pseudo_id, pred_logits = outputs
        feat1 = feat[:, 0].contiguous()
        feat2 = feat[:, 1].contiguous()
        prob1 = F.softmax(pred_logits[:, 0].contiguous(), dim=1)
        prob2 = F.softmax(pred_logits[:, 1].contiguous(), dim=1)
        rampup_scale = rampup(iter, self._cfg.UNSUPERVISED.RAMPUP_ITER, self._cfg.UNSUPERVISED.RAMPUP_COEFF)
        loss_dict = {'consistency_loss': F.mse_loss(prob1, prob2) * rampup_scale}
        loss_dict.update(RankStatLoss(self._cfg)(feat1, feat2, prob1, prob2))
        return loss_dict
        # return NTXentLoss(self._cfg)(feat1, feat2, pseudo_id)
