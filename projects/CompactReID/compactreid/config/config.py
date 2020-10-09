# encoding: utf-8
"""
@author:  wuyiming
@contact: yimingwu@hotmail.com
"""
from fvcore.common.config import CfgNode as CN


def add_compactreid_config(cfg):
    _C = cfg
    # ----------------------------------------------------------------------------
    # Unsupervised
    # ----------------------------------------------------------------------------
    _C.COMPACT = CN()
    _C.COMPACT.BLOCK = 4
    _C.COMPACT.TIMES = 1
    _C.COMPACT.LOSS_ENABLE = True
    _C.COMPACT.LOSS_SCALE = 1.

    return _C