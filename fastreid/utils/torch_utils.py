# encoding: utf-8
"""
@author:  xingyu liao
@contact: liaoxingyu5@jd.com
"""

import time
import datetime
import math
import torch
import logging
import numpy as np
from torch import nn
import torch.nn.functional as F

from . import comm
from .logger import log_every_n_seconds
from fastreid.evaluation.evaluator import inference_context


logger = logging.getLogger(__name__)

__all__ = [
    'weights_init_classifier',
    'weights_init_kaiming',
    'to_numpy',
    'to_torch',
    'tensor2im',
    'extract_features'
]


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("Arcface") != -1 or classname.find("Circle") != -1:
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray


def tensor2im(input_image, mean=0.5, std=0.5, imtype=np.uint8):
    """
    Converts a Tensor array into a numpy image array.
    :param input_image: (tensor or ndarray)
    :param mean: float or list
    :param std: float or list
    :param imtype: default is np.uint8
    :return:
    """
    if isinstance(mean, list):
        mean = np.array(mean)
    if isinstance(std, list):
        std = np.array(std)

    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy,
                                    (1, 2, 0)) * std + mean) * 255.0
    else:
        image_numpy = input_image
    image_numpy = image_numpy.clip(0, 255)
    return image_numpy.astype(imtype)


@torch.no_grad()
def extract_features(model, data_loader, norm_feat=True):
    total = len(data_loader)
    data_iter = iter(data_loader)
    start_time = time.perf_counter()
    total_compute_time = 0

    features = list()
    true_label = list()
    with inference_context(model), torch.no_grad():
        for idx in range(total):
            inputs = next(data_iter)

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if norm_feat:
                outputs = F.normalize(outputs, p=2, dim=1)
            features.append(outputs.cpu())
            true_label.append(inputs['targets'])
            comm.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            idx += 1
            iters_after_start = idx + 1
            seconds_per_img = total_compute_time / iters_after_start
            if seconds_per_img > 30:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=30,
                )

    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device)".format(
            total_time_str, total_time / total
        )
    )

    if comm.get_world_size() > 1:
        comm.synchronize()
        features = torch.cat(features)
        true_label = torch.cat(true_label)
        features = comm.all_gather(features)
        true_label = comm.all_gather(true_label)
    features = torch.cat(features, dim=0)
    true_label = torch.cat(true_label, dim=0)

    return features, true_label
