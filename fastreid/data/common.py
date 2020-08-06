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
            # Sort the list, or the set will result in different pids on different process
            self.pids = sorted(list(set([item[1] for item in img_items])))
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
        # Add the pseudo labels for un/semi-supervised learning
        self.pseudo_labels = list(range(len(self.img_items)))

    def renew_pseudo_labels(self, pseduo_labels):
        self.pseudo_labels = pseduo_labels

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
            "index": index,
            "pseudo_id": self.pseudo_labels[index]
        }

    def __getitem__(self, index):
        if isinstance(index, (tuple, list)):
            return [self._get_single_item(ind) for ind in index]
        return self._get_single_item(index)

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_pseudo_classes(self):
        return len(set(self.pseudo_labels))
