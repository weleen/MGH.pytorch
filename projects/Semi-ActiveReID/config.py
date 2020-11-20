'''
Author: WuYiming
Date: 2020-10-27 10:18:25
LastEditTime: 2020-11-17 10:16:19
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /fast-reid/projects/ActiveReID/config.py
'''
# encoding: utf-8
"""
@author:  wuyiming
@contact: yimingwu@hotmail.com
"""
from fvcore.common.config import CfgNode as CN


def add_activereid_config(cfg):
    _C = cfg
    # ----------------------------------------------------------------------------
    # Active
    # ----------------------------------------------------------------------------
    _C.ACTIVE = CN()

    # hyperparam for active learning
    _C.ACTIVE.START_ITER = 20
    _C.ACTIVE.END_ITER = 160
    _C.ACTIVE.SAMPLE_ITER = 1
    _C.ACTIVE.SAMPLE_M = 0.05  # how much percentage of samples could be queried.

    # Sampling method
    _C.ACTIVE.SAMPLER = CN()
    _C.ACTIVE.SAMPLER.QUERY_FUNC = "random"

    # Two tasks
    _C.ACTIVE.RECTIFY = True  # rectify affinity matrix for clustering
    _C.ACTIVE.BUILD_DATALOADER = True  # build dataloader with only labeled dataset

    # rectify task
    _C.ACTIVE.EDGE_PROP = True
    _C.ACTIVE.EDGE_PROP_MAX_ITER = 10

    # build active data loader
    _C.ACTIVE.NODE_PROP = True
    _C.ACTIVE.NODE_PROP_K = -1