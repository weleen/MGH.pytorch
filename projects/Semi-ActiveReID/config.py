# encoding: utf-8
"""
@author:  wuyiming
@contact: yimingwu@hotmail.com
"""
from fvcore.common.config import CfgNode as CN


def add_activereid_config(cfg):
    _C = cfg
    # Pseudo
    _C.PSEUDO.INDEP_THRE = [0.6000000238418579,] # market 0.6000000238418579 duke 0.7647058963775635 msmt 0.6000000238418579, in log, print 1-indep_thre
    # ----------------------------------------------------------------------------
    # Active
    # ----------------------------------------------------------------------------
    _C.ACTIVE = CN()
    
    # sampling cost limitation
    _C.ACTIVE.SAMPLE_M = 0.05
    _C.ACTIVE.QUERY_FUNC = "RandomSampler"

    # hyperparam for active learning
    _C.ACTIVE.START_EPOCH = 0
    _C.ACTIVE.END_EPOCH = 50
    _C.ACTIVE.SAMPLE_EPOCH = 2

    # Two tasks
    _C.ACTIVE.RECTIFY = True  # rectify affinity matrix for clustering
    _C.ACTIVE.BUILD_DATALOADER = True  # build dataloader with only labeled dataset

    # rectify task
    _C.ACTIVE.EDGE_PROP = True
    _C.ACTIVE.EDGE_PROP_METHOD = 'res' # 'cdp'
    # res
    _C.ACTIVE.EDGE_PROP_STEP = 10
    # cdp
    _C.ACTIVE.EDGE_PROP_STEP_CDP = 0.05

    # build active data loader
    _C.ACTIVE.IMS_PER_BATCH = 63
    # _C.ACTIVE.NODE_PROP = True
    # _C.ACTIVE.NODE_PROP_STEP = -1
