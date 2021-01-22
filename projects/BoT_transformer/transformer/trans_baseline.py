# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging
import torch
from torch import nn
import torch.nn.functional as F

from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from fastreid.modeling.meta_arch.build import META_ARCH_REGISTRY

from .transformer import Transformer

@META_ARCH_REGISTRY.register()
class Trans_Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        self.num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        # backbone
        self.backbone = build_backbone(cfg)

        # head
        self.heads = build_heads(cfg)
        feat_dim = cfg.MODEL.BACKBONE.FEAT_DIM
        self.transformer = Transformer(feat_dim)
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        self.pos_embedding = nn.Parameter(torch.randn(1, batch_size, feat_dim))
        self.mlp = nn.Sequential(nn.Linear(feat_dim, 1), nn.Sigmoid())
        self.bce_loss = nn.BCELoss()

    @property
    def device(self):
        return self.backbone.conv1.weight.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)
        if self.transformer.training and self.mlp.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"].long().to(self.device)

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(features, targets)
            scores, targets = self.transformer_inference(outputs, outputs, targets)
            
            return {
                "outputs": scores.view(-1),
                "targets": targets.view(-1),
            }
        else:
            outputs = self.heads(features)
            return outputs
    
    def transformer_inference(self, q_feats, g_feats, targets=None):
        max_rank = 50
        if targets is None:
            q_feats = self.preprocess_image(q_feats)
            g_feats = self.preprocess_image(g_feats)
        if self._cfg.TEST.METRIC == "cosine":
            q_feats = F.normalize(q_feats, dim=1)
            g_feats = F.normalize(g_feats, dim=1)
        distmat = cal_dist(self._cfg.TEST.METRIC, q_feats, g_feats)
        indx = torch.argsort(distmat, dim=1)
        if targets is None:
            indx = indx[:, :max_rank]
        g_feats = g_feats[indx]
        # residual
        # g_feats = g_feats - q_feats.unsqueeze(1)
        g_feats = g_feats + self.pos_embedding[:, :g_feats.size(1)]
        outputs = self.transformer(g_feats)
        scores = self.mlp(outputs)
        if targets is not None:
            targets = targets[indx]
            targets = (targets==targets[:, 0].view(-1, 1)).float()
            return scores, targets
        return scores.squeeze(2), indx

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
        return {'bce_loss': self.bce_loss(outs["outputs"], outs["targets"])}

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

def cal_dist(metric: str, query_feat: torch.tensor, gallery_feat: torch.tensor):
    assert metric in ["cosine", "euclidean"], "must choose from [cosine, euclidean], but got {}".format(metric)
    if metric == "cosine":
        dist = 1 - torch.mm(query_feat, gallery_feat.t())
    else:
        m, n = query_feat.size(0), gallery_feat.size(0)
        xx = torch.pow(query_feat, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(gallery_feat, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(query_feat, gallery_feat.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist