# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

__all__ = ['RandomPatch', 'AugMix', 'GaussianBlur', 'MutualTransform']

import copy
import math
import random
from collections import deque

import numpy as np
from PIL import Image, ImageFilter

from .functional import augmentations


class RandomPatch(object):
    """Random patch data augmentation.
    There is a patch pool that stores randomly extracted pathces from person images.
    For each input image, RandomPatch
        1) extracts a random patch and stores the patch in the patch pool;
        2) randomly selects a patch from the patch pool and pastes it on the
           input (at random position) to simulate occlusion.
    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. arXiv preprint, 2019.
    """

    def __init__(self, prob_happen=0.5, pool_capacity=50000, min_sample_size=100,
                 patch_min_area=0.01, patch_max_area=0.5, patch_min_ratio=0.1,
                 prob_rotate=0.5, prob_flip_leftright=0.5,
                 ):
        self.prob_happen = prob_happen

        self.patch_min_area = patch_min_area
        self.patch_max_area = patch_max_area
        self.patch_min_ratio = patch_min_ratio

        self.prob_rotate = prob_rotate
        self.prob_flip_leftright = prob_flip_leftright

        self.patchpool = deque(maxlen=pool_capacity)
        self.min_sample_size = min_sample_size

    def generate_wh(self, W, H):
        area = W * H
        for attempt in range(100):
            target_area = random.uniform(self.patch_min_area, self.patch_max_area) * area
            aspect_ratio = random.uniform(self.patch_min_ratio, 1. / self.patch_min_ratio)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < W and h < H:
                return w, h
        return None, None

    def transform_patch(self, patch):
        if random.uniform(0, 1) > self.prob_flip_leftright:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
        if random.uniform(0, 1) > self.prob_rotate:
            patch = patch.rotate(random.randint(-10, 10))
        return patch

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))

        W, H = img.size  # original image size

        # collect new patch
        w, h = self.generate_wh(W, H)
        if w is not None and h is not None:
            x1 = random.randint(0, W - w)
            y1 = random.randint(0, H - h)
            new_patch = img.crop((x1, y1, x1 + w, y1 + h))
            self.patchpool.append(new_patch)

        if len(self.patchpool) < self.min_sample_size:
            return img

        if random.uniform(0, 1) > self.prob_happen:
            return img

        # paste a randomly selected patch on a random position
        patch = random.sample(self.patchpool, 1)[0]
        patchW, patchH = patch.size
        x1 = random.randint(0, W - patchW)
        y1 = random.randint(0, H - patchH)
        patch = self.transform_patch(patch)
        img.paste(patch, (x1, y1))

        return img


class AugMix(object):
    """ Perform AugMix augmentation and compute mixture.
    Args:
        prob: Probability of taking augmix
        aug_prob_coeff: Probability distribution coefficients.
        mixture_width: Number of augmentation chains to mix per augmented example.
        mixture_depth: Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]'
        aug_severity: Severity of underlying augmentation operators (between 1 to 10).
    """

    def __init__(self, prob=0.5, aug_prob_coeff=0.1, mixture_width=3, mixture_depth=1, aug_severity=1):
        self.prob = prob
        self.aug_prob_coeff = aug_prob_coeff
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth
        self.aug_severity = aug_severity
        self.augmentations = augmentations

    def __call__(self, image):
        """Perform AugMix augmentations and compute mixture.
        Returns:
          mixed: Augmented and mixed image.
        """
        if random.random() > self.prob:
            return np.asarray(image)

        ws = np.float32(
            np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
        m = np.float32(np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff))

        # image = np.asarray(image, dtype=np.float32).copy()
        # mix = np.zeros_like(image)
        mix = np.zeros([image.size[1], image.size[0], 3])
        # h, w = image.shape[0], image.shape[1]
        for i in range(self.mixture_width):
            image_aug = image.copy()
            # image_aug = Image.fromarray(image.copy().astype(np.uint8))
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(self.augmentations)
                image_aug = op(image_aug, self.aug_severity)
            mix += ws[i] * np.asarray(image_aug)

        mixed = (1 - m) * image + m * mix
        return mixed.astype(np.uint8)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, img):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img

    def __repr__(self):
        return "GaussianBlur Transformer"


class MutualTransform:
    """Apply the transformer more times on a same raw image."""

    def __init__(self, transformer, times=2):
        self.transformer = transformer
        self.times = times

    def __call__(self, img):
        imgs = []
        for _ in range(self.times):
            img_copy = copy.deepcopy(img)
            img_copy = self.transformer(img_copy)
            imgs.append(img_copy)

        return imgs

    def __repr__(self):
        return "Mutual Transformer"

class RandomGrayScale:
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image):
        pass

class RandomPatchGrayScale:    
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image):
        pass