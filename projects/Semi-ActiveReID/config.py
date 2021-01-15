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
    
    # sampling instance, pair, or triplet
    _C.ACTIVE.SAMPLING_METHOD = 'instance'

    # hyperparam for active learning
    _C.ACTIVE.START_EPOCH = 20
    _C.ACTIVE.END_EPOCH = 160
    _C.ACTIVE.SAMPLE_EPOCH = 1

    # Two tasks
    _C.ACTIVE.RECTIFY = True  # rectify affinity matrix for clustering
    _C.ACTIVE.BUILD_DATALOADER = True  # build dataloader with only labeled dataset

    # rectify task
    _C.ACTIVE.EDGE_PROP = True
    _C.ACTIVE.EDGE_PROP_STEP = 10

    # build active data loader
    _C.ACTIVE.NODE_PROP = True
    _C.ACTIVE.NODE_PROP_STEP = -1
    
    # instance based sampling specific
    _C.ACTIVE.INSTANCE = CN()
    _C.ACTIVE.INSTANCE.SAMPLE_M = 0.05
    _C.ACTIVE.INSTANCE.QUERY_FUNC = "random"

    # pair based sampling specific
    _C.ACTIVE.PAIRED = CN()
    _C.ACTIVE.PAIRED.SAMPLE_M = 0.05
    _C.ACTIVE.PAIRED.QUERY_FUNC = "random"

    # triplet based sampling specific
    _C.ACTIVE.TRIPLET = CN()
    _C.ACTIVE.TRIPLET.SAMPLE_M = 0.05
    _C.ACTIVE.TRIPLET.QUERY_FUNC = "random"

