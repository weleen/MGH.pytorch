# encoding: utf-8
"""
@author:  wuyiming
@contact: yimingwu@hotmail.com
"""
from .config import add_spclreid_config
from .spclreid_trainer import SPCLTrainer
from .spclreid_baseline import USL_Baseline
from . import hooks