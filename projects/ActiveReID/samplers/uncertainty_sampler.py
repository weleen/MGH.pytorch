# encoding: utf-8
"""
@author:  tianjian
"""

from .build import ACTIVE_SAMPLERS_REGISTRY


@ACTIVE_SAMPLERS_REGISTRY.register()
class UncertaintySampler:
    def __init__(self, cfg):
        pass

    def sample(self, model):
        # model inference

        # calculate uncertainty

        # sort

        # select samples

        # update labeled and unlabeled set

        pass
