# encoding: utf-8
"""
@author:  wuyiming
@contact: yimingwu@hotmail.com
"""
from fvcore.common.config import CfgNode as CN


def add_ssc_config(cfg):
    _C = cfg

    # ----------------------------------------------------------------------------
    # Unsupervised
    # ----------------------------------------------------------------------------
    _C.UNSUPERVISED = CN()
    # Number of augmented images
    _C.UNSUPERVISED.AUG_K = 2
    # NTXent loss
    _C.UNSUPERVISED.LOSS = CN()
    # Temperature for ntxent loss
    _C.UNSUPERVISED.LOSS.TEMP = 0.1
    # Clustering iteration
    _C.UNSUPERVISED.CLUSTER_ITER = 400
    # Number of clusters
    _C.UNSUPERVISED.CLUSTER_SIZE = 500
    # Topk for RankStatisticLoss
    _C.UNSUPERVISED.TOPK = 5
    _C.UNSUPERVISED.RAMPUP_ITER = 50
    _C.UNSUPERVISED.RAMPUP_COEFF = 10.
