# encoding: utf-8
"""
@author:  xingyu liao
@contact: liaoxingyu5@jd.com
"""
import os
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


def tensor2im(input_image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), imtype=np.uint8):
    """
    Converts a Tensor array into a numpy image array.
    :param input_image: (tensor or ndarray)
    :param mean: float or list
    :param std: float or list
    :param imtype: default is np.uint8
    :return:
    """
    if isinstance(mean, (list, tuple)):
        mean = np.array(mean)
    if isinstance(std, (list, tuple)):
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
def extract_features(model, data_loader, norm_feat=True, save_path=None):
    """
    features in output maybe [N, K, C] or [N, C], where K is not 1 when MultiPartHead is enabled.
    """
    total = len(data_loader)
    data_iter = iter(data_loader)
    start_time = time.perf_counter()
    total_compute_time = 0
    if save_path is not None:
        file_name = '{}_feature_label.pt'.format(save_path)
        img_path_file = '{}_img_path.txt'.format(save_path)
        if os.path.exists(file_name) and os.path.exists(img_path_file):
            res = torch.load(file_name)
            features, true_label, camids, indexes = res['features'], res['true_label'], res['camids'], res['indexes']
            with open(img_path_file, 'r') as f:
                img_paths = [path.strip('\n') for path in f.readlines()]
            return features, true_label, img_paths, camids, indexes

    features = list()
    true_label = list()
    img_paths = list()
    camids = list()
    indexes = list()
    with inference_context(model), torch.no_grad():
        for idx in range(total):
            inputs = next(data_iter)

            start_compute_time = time.perf_counter()
            assert 'images' in inputs, 'images not found in inputs, only {}'.format(inputs.keys())
            if isinstance(inputs['images'], list): # inputs is a dict, 'images' must be in inputs
                input_dict = {}
                outputs = []
                for i in range(len(inputs['images'])):
                    for k, v in inputs.items():
                        input_dict[k] = v[i] if k == 'images' else v
                    outputs.append(model(input))
                outputs = torch.stack(outputs, dim=0).mean(dim=0)
            else:
                outputs = model(inputs)
            if norm_feat:
                if isinstance(outputs, list):
                    outputs = torch.stack([F.normalize(output, p=2, dim=1) for output in outputs], dim=1)
                else:
                    outputs = F.normalize(outputs, p=2, dim=1).unsqueeze(1)
            features.append(outputs.data.cpu())
            true_label.append(inputs['targets'])
            camids.append(inputs['camids'])
            indexes.append(inputs['index'])
            img_paths.extend(inputs['img_paths'])
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
        camids = torch.cat(camids)
        indexes = torch.cat(indexes)
        features = comm.all_gather(features)
        true_label = comm.all_gather(true_label)
        camids = comm.all_gather(camids)
        indexes = comm.all_gather(indexes)
        img_paths = comm.all_gather(img_paths)
        img_paths = sum(img_paths, [])
    features = torch.cat(features, dim=0)
    if features.size(1) == 1:
        # compactible with the original single output from model
        features = features.squeeze(1)
    true_label = torch.cat(true_label, dim=0)
    camids = torch.cat(camids, dim=0)
    indexes = torch.cat(indexes, dim=0)

    if comm.is_main_process():
        if save_path is not None:
            file_name = '{}_feature_label.pt'.format(save_path)
            img_path_file = '{}_img_path.txt'.format(save_path)
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            res = {'features': features, 'true_label': true_label, 'camids': camids, 'indexes': indexes}
            torch.save(res, file_name)
            with open(img_path_file, 'w') as f:
                for path in img_paths:
                    f.write(path + '\n')

    return features, true_label, img_paths, camids, indexes
