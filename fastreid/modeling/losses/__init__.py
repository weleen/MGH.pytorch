# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

from .build_losses import reid_losses
from .cross_entroy_loss import CrossEntropyLoss, SoftEntropyLoss, CenterContrastiveLoss, HardViewContrastiveLoss
from .focal_loss import FocalLoss
from .triplet_loss import TripletLoss, SoftmaxTripletLoss, SoftSoftmaxTripletLoss, ActiveTripletLoss
from .circle_loss import CircleLoss
from .pairwise_smooth_loss import PairwiseSmoothLoss
from .hybrid_memory import HybridMemory