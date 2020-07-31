# encoding: utf-8
"""
@author:  wuyiming
@contact: yimingwu@hotmail.com
"""
import copy
import torch
from torch.utils.data import Dataset

from fastreid.utils.misc import read_image


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True, cfg=None):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel
        self._cfg = cfg

        self.pid_dict = {}
        if self.relabel:
            self.pids = list(set([item[1] for item in img_items]))
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])

    def __len__(self):
        return len(self.img_items)

    def _get_single_item(self, index):
        img_path, pid, camid = self.img_items[index]
        img = read_image(img_path)
        # run augmentation several times
        imgs = list()
        if self.transform is not None:
            for i in range(self._cfg.SELFSUP.AUG_K):
                imgs.append(self.transform(img))
        imgs = torch.cat(imgs, 0)
        if self.relabel: pid = self.pid_dict[pid]
        return {
            "images": imgs,
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
