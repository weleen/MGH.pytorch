# encoding: utf-8
"""
@author:  wuyiming
@contact: yimingwu@hotmail.com
"""
from fvcore.common.config import CfgNode as CN


def add_advcluster_config(cfg):
    _C = cfg

    # ----------------------------------------------------------------------------
    # Unsupervised
    # ----------------------------------------------------------------------------
    _C.ADVCLUSTER = CN()
    # DBSCAN parameter
    _C.ADVCLUSTER.CLUSTER_METHODS = ['kmeans', 'DBSCAN']
    _C.ADVCLUSTER.CLUSTER_PARAMS = [500, 0.4]
    _C.ADVCLUSTER.CLUSTER_PERIOD = 2
