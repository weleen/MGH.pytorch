# encoding: utf-8
"""
@author:  wuyiming
@contact: yimingwu@hotmail.com
"""
from fvcore.common.config import CfgNode as CN


def add_ssc_config(cfg):
    _C = cfg

    # -----------------------------------------------------------------------------
    # DataLoader
    # -----------------------------------------------------------------------------
    _C.DATALOADER.PK_SAMPLER = False
    _C.DATALOADER.SAMPLER_NAME = "TrainingSampler"

    # ----------------------------------------------------------------------------
    # Self-supervised, stage 1
    # ----------------------------------------------------------------------------
    _C.SELFSUP = CN()
    # number of augmented images
    _C.SELFSUP.AUG_K = 2

    _C.SELFSUP.SOLVER = CN()
    _C.SELFSUP.SOLVER.NAMES = ("BCELoss")

    # ----------------------------------------------------------------------------
    # Unsupervised, stage 2
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
