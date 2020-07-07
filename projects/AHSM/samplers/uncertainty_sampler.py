# encoding: utf-8
"""
@author:  tianjian
"""

from .build import ACTIVE_SAMPLERS_REGISTRY


@ACTIVE_SAMPLERS_REGISTRY.register()
class UncertaintySampler:
    def __init__(self, labeled_set, unlabeled_set, labeled_num):
        self.labeled_set = labeled_set
        self.unlabeled_set = unlabeled_set
        self.labeled_num = labeled_num

    def sample(self, model):
        # model inference

        # calculate uncertainty

        # sort

        # select samples

        # update labeled and unlabeled set

        pass