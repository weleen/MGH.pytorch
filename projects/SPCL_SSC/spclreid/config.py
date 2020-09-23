# encoding: utf-8
"""
@author:  wuyiming
@contact: yimingwu@hotmail.com
"""
from fvcore.common.config import CfgNode as CN


def add_spclreid_config(cfg):
    _C = cfg
    # ----------------------------------------------------------------------------
    # Unsupervised
    # ----------------------------------------------------------------------------
    _C.UNSUPERVISED = CN()
    # DBSCAN parameter
    _C.UNSUPERVISED.EPS = 0.6
    _C.UNSUPERVISED.EPS_GAP = 0.02
    _C.UNSUPERVISED.CLUSTER_ITER = 400
    # Memory related options for Self-paced learning
    # Temperature for scaling contrastive loss
    _C.UNSUPERVISED.MEMORY_TEMP = 0.05
    # Update momentum for the hybrid memory
    _C.UNSUPERVISED.MEMORY_MOMENTUM = 0.2
    # Reset Optimizer
    _C.UNSUPERVISED.RESET_OPT = False
    # Number of clusters
    _C.UNSUPERVISED.CLUSTER_SIZE = 500
    # Number of augmented images
    _C.UNSUPERVISED.AUG_K = 2
    # Topk for RankStatisticLoss
    _C.UNSUPERVISED.TOPK = 5
    _C.UNSUPERVISED.RAMPUP_ITER = 50
    _C.UNSUPERVISED.RAMPUP_COEFF = 10.
