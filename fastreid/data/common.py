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
        self.transform = transform

        if relabel:
            # Please relabel the dataset in the first epoch
            start_pid = 0
            start_camid = 0
            for data_ind, dataset in enumerate(self.datasets):
                pids = sorted(list(set([d[1] for d in dataset.data])))
                cams = sorted(list(set([d[2] for d in dataset.data])))
                pid_dict = dict([(p, i) for i, p in enumerate(pids)])
                cam_dict = dict([(p, i) for i, p in enumerate(cams)])
                for idx, data in enumerate(dataset.data):
                    new_data = (data[0], pid_dict[data[1]] + start_pid, cam_dict[data[2]] + start_camid)
                    self.datasets[data_ind].data[idx] = new_data
                added_pid, added_camid = dataset.parse_data(self.datasets[data_ind].data)
                start_pid += added_pid
                start_camid += added_camid
                self.img_items.extend(self.datasets[data_ind].data)
        else:
            for dataset in self.datasets:
                self.img_items.extend(dataset.data)


    @property
    def datasets_size(self):
        return [len(dataset.data) for dataset in self.datasets]

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path, pid, camid = self.img_items[index]
        img = read_image(img_path)
        if self.transform is not None: img = self.transform(img)
        return {
            "images": img,
            "targets": int(pid),
            "camids": camid,
            "img_paths": img_path,
            "index": int(index)
        }

    @property
    def num_classes(self):
        class_set = set([item[1] for item in self.img_items])
        if -1 in class_set: return len(class_set) - 1
        return len(class_set)

    @property
    def num_cameras(self):
        return len(set([item[2] for item in self.img_items]))
