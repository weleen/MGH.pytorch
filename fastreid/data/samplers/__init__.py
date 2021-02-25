# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from fvcore.common.registry import Registry

SAMPLER_REGISTRY = Registry("SAMPLER")
SAMPLER_REGISTRY.__doc__ = """
Registry for sampling strategy.
It must returns an instance of :class:Sampler
"""

from .triplet_sampler import BalancedIdentitySampler, NaiveIdentitySampler, RandomMultipleGallerySampler, ProxyBalancedSampler, ProxySampler
from .data_sampler import TrainingSampler, InferenceSampler
from .active_triplet_sampler import ActiveTripletSampler

__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
