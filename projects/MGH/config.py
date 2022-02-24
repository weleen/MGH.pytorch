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
    _C.CAP.NORM_FEAT = True
    _C.CAP.TEMP = 0.07
    _C.CAP.ENABLE_HARD_NEG = True
    _C.CAP.HARD_NEG_K = 50
    _C.CAP.MOMENTUM = 0.2
    _C.CAP.LOSS_WEIGHT = 0.5
    _C.CAP.INTERCAM_EPOCH = 5

    _C.CAP.ST_TEST = False
    _C.CAP.INSTANCE_LOSS = False

    _C.CAP.WEIGHTED_INTRA = False
    _C.CAP.WEIGHTED_INTER = False

    _C.CAP.LOSS_IDENTITY = CN()
    _C.CAP.LOSS_IDENTITY.SCALE = 0.5
    _C.CAP.LOSS_IDENTITY.START_EPOCH = 5
    _C.CAP.LOSS_IDENTITY.WEIGHTED = False

    _C.CAP.LOSS_CAMERA = CN()
    _C.CAP.LOSS_CAMERA.SCALE = 1.0
    _C.CAP.LOSS_CAMERA.START_EPOCH = 0
    _C.CAP.LOSS_CAMERA.WEIGHTED = False

    _C.CAP.LOSS_INSTANCE = CN()
    _C.CAP.LOSS_INSTANCE.SCALE = 1.0
    _C.CAP.LOSS_INSTANCE.START_EPOCH = 5
    _C.CAP.LOSS_INSTANCE.LIST_LENGTH = 1000
    _C.CAP.LOSS_INSTANCE.NUM_BINS = 25
    _C.CAP.LOSS_INSTANCE.THRESH = 0.1
    _C.CAP.LOSS_INSTANCE.NAME = 'aploss'  # 'smoothaploss'
    _C.CAP.LOSS_INSTANCE.SMOOTHAP_TARGET = False
    _C.CAP.LOSS_INSTANCE.MOMENTUM = 1.0
