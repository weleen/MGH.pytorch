# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "BatchNorm",
    "IBN",
    "GhostBatchNorm",
    "FrozenBatchNorm",
    "SyncBatchNorm",
    "DSBatchNorm",
    "get_norm",
    "convert_dsbn",
    "convert_sync_bn"
]


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0,
                 bias_init=0.0, **kwargs):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None: nn.init.constant_(self.weight, weight_init)
        if bias_init is not None: nn.init.constant_(self.bias, bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)


class SyncBatchNorm(nn.SyncBatchNorm):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0,
                 bias_init=0.0):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None: nn.init.constant_(self.weight, weight_init)
        if bias_init is not None: nn.init.constant_(self.bias, bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)


class IBN(nn.Module):
    def __init__(self, planes, bn_norm, **kwargs):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = get_norm(bn_norm, half2, **kwargs)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits=1, **kwargs):
        super().__init__(num_features, **kwargs)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            self.running_mean = self.running_mean.repeat(self.num_splits)
            self.running_var = self.running_var.repeat(self.num_splits)
            outputs = F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0)
            return outputs
        else:
            return F.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps)


class FrozenBatchNorm(BatchNorm):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.
    The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
    which are computed from the original four parameters of BN.
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
    When loading a backbone model from Caffe2, "running_mean" and "running_var"
    will be left unchanged as identity transformation.
    Other pre-trained backbone models may contain all 4 parameters.
    The forward is implemented by `F.batch_norm(..., training=False)`.
    """

    _version = 3

    def __init__(self, num_features, eps=1e-5, **kwargs):
        super().__init__(num_features, weight_freeze=True, bias_freeze=True, **kwargs)
        self.num_features = num_features
        self.eps = eps

    def forward(self, x):
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as well.
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            return x * scale + bias
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # No running_mean/var in early versions
            # This will silent the warnings
            if prefix + "running_mean" not in state_dict:
                state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
            if prefix + "running_var" not in state_dict:
                state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

        if version is not None and version < 3:
            logger = logging.getLogger(__name__)
            logger.info("FrozenBatchNorm {} is upgraded to version 3.".format(prefix.rstrip(".")))
            # In version < 3, running_var are used without +eps.
            state_dict[prefix + "running_var"] -= self.eps

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.
        Args:
            module (torch.nn.Module):
        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.
        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


class DSBatchNorm(nn.Module):
    def __init__(
        self,
        num_features,
        num_domains,
        batchnorm_layer=nn.BatchNorm2d,
        eps=1e-5,
        momentum=0.1,
        target_bn_idx=-1,
        weight_requires_grad=True,
        bias_requires_grad=True,
    ):
        super(DSBatchNorm, self).__init__()
        self.num_features = num_features
        self.num_domains = num_domains
        self.target_bn_idx = target_bn_idx
        self.batchnorm_layer = batchnorm_layer

        dsbn = [batchnorm_layer(num_features, eps=eps, momentum=momentum) 
                    for _ in range(num_domains)]
        for idx in range(num_domains):
            dsbn[idx].weight.requires_grad_(weight_requires_grad)
            dsbn[idx].bias.requires_grad_(bias_requires_grad)
        self.dsbn = nn.ModuleList(dsbn)

    def forward(self, x):
        if self.training:
            return self._forward_train(x)
        else:
            return self._forward_test(x)

    def _forward_train(self, x):
        bs = x.size(0)
        assert bs % self.num_domains == 0, "the batch size should be times of BN groups"

        split = torch.split(x, int(bs // self.num_domains), 0)
        out = []
        for idx, subx in enumerate(split):
            out.append(self.dsbn[idx](subx.contiguous()))
        return torch.cat(out, 0)

    def _forward_test(self, x):
        # Default: the last BN is adopted for target domain
        return self.dsbn[self.target_bn_idx](x)


def get_norm(norm, out_channels, **kwargs):
    """
    Args:
        norm (str or callable): either one of BN, GhostBN, FrozenBN, GN or SyncBN;
            or a callable that thakes a channel number and returns
            the normalization layer as a nn.Module
        out_channels: number of channels for normalization layer

    Returns:
        nn.Module or None: the normalization layer
    """
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": BatchNorm,
            "GhostBN": GhostBatchNorm,
            "FrozenBN": FrozenBatchNorm,
            "GN": lambda channels, **args: nn.GroupNorm(32, channels),
            "syncBN": SyncBatchNorm,
        }[norm]
    return norm(out_channels, **kwargs)


def convert_dsbn(model, num_domains=2, target_bn_idx=-1):
    """
    convert all bn layers in the model to domain-specific bn layers
    """
    for _, (child_name, child) in enumerate(model.named_children()):
        if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d)):
            class_type = type(child)
            # BN 1/2d -> DSBN 1/2d
            m = DSBatchNorm(
                child.num_features,
                num_domains,
                class_type,
                child.eps,
                child.momentum,
                target_bn_idx,
                child.weight.requires_grad,
                child.bias.requires_grad,
            )
            m.to(next(child.parameters()).device)

            for idx in range(num_domains):
                m.dsbn[idx].load_state_dict(child.state_dict())

            setattr(model, child_name, m)
        else:
            # recursive searching
            convert_dsbn(child, num_domains=num_domains, target_bn_idx=target_bn_idx)


def convert_sync_bn(model, process_group=None):
    for _, (child_name, child) in enumerate(model.named_children()):
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            if isinstance(child, nn.modules.instancenorm._InstanceNorm):
                continue
            m = nn.SyncBatchNorm.convert_sync_batchnorm(child, process_group)
            m.weight.requires_grad_(child.weight.requires_grad)
            m.bias.requires_grad_(child.bias.requires_grad)
            m.to(next(child.parameters()).device)
            setattr(model, child_name, m)
        else:
            convert_sync_bn(child, process_group)