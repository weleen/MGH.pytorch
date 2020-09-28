# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import Dataset

from fastreid.utils.misc import read_image


class CommDataset(Dataset):
    """compatible with un/semi-supervised learning"""

    def __init__(self, datasets, transform=None, relabel=True):
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
