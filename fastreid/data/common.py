# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import Dataset

from fastreid.utils.misc import read_image


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

        pid_set = set([i[1] for i in img_items])

        self.pids = sorted(list(pid_set))
        if relabel: self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
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

    @property
    def num_classes(self):
        return len(self.pids)


class NewCommDataset(Dataset):
    """compatible with un/semi-supervised learning"""

    def __init__(self, datasets, transform=None, relabel: bool = True):
        if isinstance(datasets, list):
            self.datasets = datasets  # add this property to save the original dataset information
        else:
            self.datasets = [datasets]
        self.img_items = list()
        for dataset in self.datasets:
            self.img_items.extend(dataset.data)
        self.transform = transform
        self.relabel = relabel

        pid_set = set([i[1] for i in self.img_items])

        self.pids = sorted(list(pid_set))
        if relabel: self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])


    @property
    def datasets_size(self):
        return [len(dataset.data) for dataset in self.datasets]

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path, pid, camid = self.img_items[index]
        img = read_image(img_path)
        if self.transform is not None: img = self.transform(img)
        if self.relabel:
            pid = self.pid_dict[pid]
        return {
            "images": img,
            "targets": pid,
            "camid": camid,
            "img_path": img_path,
            "index": index
        }

    @property
    def num_classes(self):
        return len(set([item[1] for item in self.img_items]))
