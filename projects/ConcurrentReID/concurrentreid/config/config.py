# encoding: utf-8
"""
@author:  wuyiming
@contact: yimingwu@hotmail.com
"""
from fvcore.common.config import CfgNode as CN


def add_concurrentreid_config(cfg):
    _C = cfg
    # ----------------------------------------------------------------------------
    # Unsupervised
    # ----------------------------------------------------------------------------
    _C.CONCURRENT = CN()

    # _C.CONCURRENT.TIMES = 1
    # _C.CONCURRENT.LOSS_ENABLE = True
    # _C.CONCURRENT.LOSS_SCALE = 1.

    return _C