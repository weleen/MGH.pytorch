# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

from .build_losses import reid_losses
from .cross_entroy_loss import CrossEntropyLoss, SoftEntropyLoss
from .focal_loss import FocalLoss
from .triplet_loss import TripletLoss, SoftmaxTripletLoss
from .circle_loss import CircleLoss
from .pairwise_smooth_loss import PairwiseSmoothLoss
