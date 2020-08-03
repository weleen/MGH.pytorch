# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch.utils.data import Dataset

from fastreid.utils.misc import read_image


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

        self.pid_dict = {}
        if self.relabel:
            self.pids = list(set([item[1] for item in img_items]))
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])

    def __len__(self):
        return len(self.img_items)

    def _get_single_item(self, index):
        img_path, pid, camid = self.img_items[index]
        img = read_image(img_path)
        if self.transform is not None: img = self.transform(img)
        if self.relabel: pid = self.pid_dict[pid]
        return {
            "images": img,
            "targets": pid,
            "camid": camid,
            "img_path": img_path,
            "index": index
        }

    def __getitem__(self, index):
        if isinstance(index, (tuple, list)):
            return [self._get_single_item(ind) for ind in index]
        return self._get_single_item(index)

    @property
    def num_classes(self):
        return len(self.pids)


class NewCommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        """
        :param img_items: list(list)
        :param transform:
        :param relabel:
        """
        self.img_items = img_items
        self.dataset = sum(self.img_items, [])
        self.transform = transform
        self.relabel = relabel

        self.pid_dict = {}
        if self.relabel:
            self.pids = list(set([item[1] for item in self.dataset]))
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])

    def __len__(self):
        return len(self.dataset)

    def _get_single_item(self, index):
        img_path, pid, camid = self.img_items[index]
        img = read_image(img_path)
        if self.transform is not None: img = self.transform(img)
        if self.relabel: pid = self.pid_dict[pid]
        return {
            "images": img,
            "targets": pid,
            "camid": camid,
            "img_path": img_path,
            "index": index
        }

    def __getitem__(self, index):
        if isinstance(index, (tuple, list)):
            return [self._get_single_item(ind) for ind in index]
        return self._get_single_item(index)

    @property
    def num_classes(self):
        return len(self.pids)
