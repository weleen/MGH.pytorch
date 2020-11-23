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
    _C.CONCURRENT.ENABLED = True
    _C.CONCURRENT.BLOCK_SIZE = (2, 2)
    _C.CONCURRENT.SHUFFLE = True

    return _C