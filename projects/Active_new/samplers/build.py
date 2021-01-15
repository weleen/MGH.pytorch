# encoding: utf-8
"""
@author:  tianjian
"""

from fvcore.common.registry import Registry

ACTIVE_SAMPLERS_REGISTRY = Registry("SAMPLERS")
ACTIVE_SAMPLERS_REGISTRY.__doc__ = """
Registry for samplers with active learning strategies.
The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`Samplers`.
"""


def build_active_samplers(cfg, index_set, **kwargs):
    """
    Build ActiveSamplers defined by `cfg.MODEL.ACTIVE.SAMPLER.NAME`.
    """
    sampler = cfg.ACTIVE.SAMPLER.NAME
    return ACTIVE_SAMPLERS_REGISTRY.get(sampler)(cfg, index_set, **kwargs)
