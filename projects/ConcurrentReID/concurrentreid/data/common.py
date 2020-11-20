# encoding: utf-8
'''
Author: WuYiming
Date: 2020-10-28 00:21:29
LastEditTime: 2020-10-28 20:35:49
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /fast-reid/projects/ConcurrentReID/concurrentreid/data/common.py
'''
from numpy.core.shape_base import block
import torch

from fastreid.data.common import CommDataset as Dataset
from fastreid.utils.misc import read_image
from fastreid.utils.torch_utils import tensor2im

class CommDataset(Dataset):
    """compatible with un/semi-supervised learning"""

    def __init__(self, datasets, transform=None, relabel=True, cfg=None):
        if isinstance(datasets, list):
            self.datasets = datasets  # add this property to save the original dataset information
        else:
            self.datasets = [datasets]
        self.img_items = list()
        for dataset in self.datasets:
            self.img_items.extend(dataset.data)
        self.transform = transform
        self.relabel = relabel

        pid_set = set()
        cam_set = set()
        for i in self.img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

        self.cfg = cfg

    @property
    def datasets_size(self):
        return [len(dataset.data) for dataset in self.datasets]

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path, pid, camid = self.img_items[index]
        img = read_image(img_path)
        if self.transform is not None: img = self.transform(img)
        if isinstance(img, list): img = torch.stack(img)
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
        return {
            "images": img,
            "targets": int(pid),
            "camids": camid,
            "img_paths": img_path,
            "index": int(index)
        }

    @property
    def num_classes(self):
        return len(set([item[1] for item in self.img_items]))

    @property
    def num_cameras(self):
        return len(set([item[2] for item in self.img_items]))


def stitch(imgs: list):
    length = len(imgs)
    block_size = int(length ** 0.5)
    assert block_size ** 2 == length
    img_ = imgs[0]
    c, h, w = img_.size()
    new_tensor_shape = (c, h * block_size, w * block_size)
    new_img = torch.zeros(new_tensor_shape).to(img_.device)
    for i in range(block_size): # height
        for j in range(block_size): # width
            x0, y0, x1, y1 = j * w, i * h, (j + 1) * w, (i + 1) * h
            new_img[:, y0:y1, x0:x1] = imgs[i * block_size + j]
    return new_img