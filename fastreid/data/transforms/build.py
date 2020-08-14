# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torchvision.transforms as T

from .transforms import *
from .autoaugment import AutoAugment


def build_transforms(cfg, is_train=True):
    res = []

    normalizer = T.Normalize(mean=cfg.MODEL.PIXEL_MEAN,
                             std=cfg.MODEL.PIXEL_STD)
    if is_train:
        size_train = cfg.INPUT.SIZE_TRAIN

        # augmix augmentation
        do_augmix = cfg.INPUT.DO_AUGMIX

        # auto augmentation
        do_autoaug = cfg.INPUT.DO_AUTOAUG
        total_iter = cfg.SOLVER.MAX_ITER

        # horizontal filp
        do_flip = cfg.INPUT.DO_FLIP
        flip_prob = cfg.INPUT.FLIP_PROB

        # padding
        do_pad = cfg.INPUT.DO_PAD
        padding = cfg.INPUT.PADDING
        padding_mode = cfg.INPUT.PADDING_MODE

        # gaussian blur
        do_blur = cfg.INPUT.DO_BLUR
        blur_prob = cfg.INPUT.BLUR_PROB

        # color jitter
        do_cj = cfg.INPUT.DO_CJ

        # random erasing
        do_rea = cfg.INPUT.REA.ENABLED
        rea_prob = cfg.INPUT.REA.PROB
        rea_mean = cfg.INPUT.REA.MEAN

        # random patch
        do_rpt = cfg.INPUT.RPT.ENABLED
        rpt_prob = cfg.INPUT.RPT.PROB

        # do mutual transform
        do_mutual = cfg.INPUT.MUTUAL.ENABLED
        mutual_times = cfg.INPUT.MUTUAL.TIMES

        if do_autoaug:
            res.append(AutoAugment(total_iter))
        res.append(T.Resize(size_train, interpolation=3))
        if do_flip:
            res.append(T.RandomHorizontalFlip(p=flip_prob))
        if do_pad:
            res.extend([T.Pad(padding, padding_mode=padding_mode),
                        T.RandomCrop(size_train)])
        if do_blur:
            res.append(T.RandomApply([GaussianBlur([0.1, 2.0])], p=blur_prob))
        if do_cj:
            res.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0))
        if do_augmix:
            res.append(AugMix())
        res.append(T.ToTensor())
        res.append(normalizer)
        if do_rea:
            res.append(RandomErasing(probability=rea_prob, mean=rea_mean))
        if do_rpt:
            res.append(RandomPatch(prob_happen=rpt_prob))
        if do_mutual:
            return MutualTransform(T.Compose(res), mutual_times)
    else:
        size_test = cfg.INPUT.SIZE_TEST
        res.append(T.Resize(size_test, interpolation=3))
        res.append(T.ToTensor())
        res.append(normalizer)
    return T.Compose(res)
