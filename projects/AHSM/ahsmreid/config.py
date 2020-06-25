# encoding: utf-8
"""
@author:  wenhuzhang
@contact: Andrew-pph@outlook.com
"""
from fastreid.config.config import CfgNode as CN


def add_ahsmreid_config(cfg):
    _C = cfg

    # ----------------------------------------------------------------------------
    # Active
    # ----------------------------------------------------------------------------
    _C.ACTIVE = CN()
    # ACTIVE parameter
    _C.ACTIVE.INITIAL_RATE = 0.001
    _C.ACTIVE.TRIALS = 1
    _C.ACTIVE.SAMPLE_CYCLES = 10
    _C.ACTIVE.TRAIN_CYCLES = 50
    


