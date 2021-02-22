# encoding: utf-8
"""
@author:  wuyiming
@contact: yimingwu@hotmail.com
"""
from fvcore.common.config import CfgNode as CN


def add_cap_config(cfg):
    _C = cfg

    # CAP
    _C.CAP = CN()
    _C.CAP.TEMP = 0.07
    _C.CAP.HARD_NEG_K = 50
    _C.CAP.MOMENTUM = 0.2
    _C.CAP.LOSS_WEIGHT = 0.5
    _C.CAP.INTERCAM_EPOCH = 5

    _C.CAP.ST_TEST = False
    _C.CAP.INSTANCE_LOSS = False
