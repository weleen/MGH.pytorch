'''
Author: WuYiming
Date: 2020-10-28 00:21:29
LastEditTime: 2020-10-28 21:22:36
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /fast-reid/projects/ConcurrentReID/concurrentreid/data/__init__.py
'''
from .common import CommDataset, stitch
from .build import build_reid_train_loader, build_reid_test_loader