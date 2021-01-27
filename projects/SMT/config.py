# encoding: utf-8
"""
@author:  wuyiming
@contact: yimingwu@hotmail.com
"""
from fvcore.common.config import CfgNode as CN


def add_smtreid_config(cfg):
    _C = cfg
    # Center Contrastive Loss
    _C.MODEL.LOSSES.CCL = CN()
    _C.MODEL.LOSSES.CCL.SCALE = 1.0
    _C.MODEL.LOSSES.CCL.TAU = 0.5
    _C.MODEL.LOSSES.CCL.NORM_FEAT = True

    # Hard View Contrastive Loss
    _C.MODEL.LOSSES.VCL = CN()
    _C.MODEL.LOSSES.VCL.SCALE = 1.0
    _C.MODEL.LOSSES.VCL.TAU = 0.09
    _C.MODEL.LOSSES.VCL.NORM_FEAT = True

    # Softmax Cross Entropy Loss options
    _C.MODEL.LOSSES.SCE = CN()
    _C.MODEL.LOSSES.SCE.SCALE = 1.0

    # Softmax Triplet Loss options
    _C.MODEL.LOSSES.STRI = CN()
    _C.MODEL.LOSSES.STRI.SCALE = 1.0
    _C.MODEL.LOSSES.STRI.NORM_FEAT = True
    _C.MODEL.LOSSES.STRI.MARGIN = 0.0

    # Soft Softmax Triplet Loss options
    _C.MODEL.LOSSES.SSTRI = CN()
    _C.MODEL.LOSSES.SSTRI.SCALE = 1.0
    _C.MODEL.LOSSES.SSTRI.NORM_FEAT = True
    _C.MODEL.LOSSES.SSTRI.MARGIN = 0.0
