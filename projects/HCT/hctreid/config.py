# encoding: utf-8
"""
@author:  tianjian
"""
from fvcore.common.config import CfgNode as CN


def add_hct_config(cfg):
    _C = cfg

    # ----------------------------------------------------------------------------
    # HCT
    # ----------------------------------------------------------------------------
    _C.HCT = CN()
    
    _C.HCT.MERGE_PERCENT = 0.07
    _C.HCT.MERGE_STEPS = 13
    _C.HCT.SIZE_PENALTY = 0.003
    # max epochs = loop * epochs_per_loop
    _C.HCT.EPOCHS_PER_LOOP = 20
