'''
Author: WuYiming
Date: 2020-10-27 10:18:25
LastEditTime: 2020-11-05 13:53:42
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

    # ACTIVE parameter
    _C.ACTIVE.START_EPOCH = 0
    _C.ACTIVE.INITIAL_RATE = 0.1
    _C.ACTIVE.TRAIN_EPOCH = 2

    # Limitation
    _C.ACTIVE.SAMPLE_K = 5
    _C.ACTIVE.SAMPLE_M = 0.5
    _C.ACTIVE.PAIR_TOP_RANK = 30

    # Sampling method
    _C.ACTIVE.SAMPLER = CN()
    _C.ACTIVE.SAMPLER.QUERY_FUNC = "entropy"
    _C.ACTIVE.SAMPLER.PAIR_FUNC = "random"

    # Two separate tasks
    _C.ACTIVE.RECTIFY = True
    _C.ACTIVE.BUILD_DATALOADER = False
