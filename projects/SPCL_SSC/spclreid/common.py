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

    def __init__(self, img_items, transform=None, relabel=True, cfg=None, is_train=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel
        self._cfg = cfg
        self.is_train = is_train

        pid_set = set([i[1] for i in img_items])

        self.pids = sorted(list(pid_set))
        if relabel: self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path, pid, camid = self.img_items[index]
        img = read_image(img_path)
        if self.relabel: pid = self.pid_dict[pid]

        imgs = list()
        if self.transform is not None:
            if self._cfg is not None and self.is_train:
                for i in range(self._cfg.UNSUPERVISED.AUG_K):
                    imgs.append(self.transform(img))
                imgs = torch.stack(imgs, 0)
            else:
                imgs = self.transform(img)
        return {
            "images": imgs,
            "targets": pid,
            "camid": camid,
            "img_path": img_path,
            "index": int(index)
        }

    @property
    def num_classes(self):
        return len(self.pids)