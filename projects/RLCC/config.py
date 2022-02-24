from fvcore.common.config import CfgNode as CN


def add_rlcc_config(cfg):
    _C = cfg
    # ----------------------------------------------------------------------------
    # RLCC
    # ----------------------------------------------------------------------------
    _C.PSEUDO.RLCC = CN()  # RLCC method, refer to the paper ``Refining Pseudo Labels with Clustering Consensus over Generations for Unsupervised Object Re-identification''
    _C.PSEUDO.RLCC.ENABLED = True
    _C.PSEUDO.RLCC.START_EPOCH = 0
