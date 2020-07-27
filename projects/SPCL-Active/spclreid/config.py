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

    # ----------------------------------------------------------------------------
    # Active
    # ----------------------------------------------------------------------------
    _C.ACTIVE = CN()
    # ACTIVE parameter
    _C.ACTIVE.INITIAL_RATE = 0.1
    _C.ACTIVE.TRAIN_CYCLES = 2000
    _C.ACTIVE.SAMPLE_K = 10
    _C.ACTIVE.IMS_PER_BATCH = 90
    # 'RandomSampler' or 'UncertaintySampler'
    _C.ACTIVE.SAMPLER = CN()
    _C.ACTIVE.SAMPLER.NAME = 'RandomSampler'
