import sys
import os
import os.path as osp

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class HWData(ImageDataset):
    dataset_usl = None
    dataset_name = "hwdata"

    def __init__(self, root="datasets", **kwargs):
        self.root = root
        self.dataset_dir = self.root

        self.train_dir = osp.join(self.dataset_dir, self.dataset_name)

        required_files = {
            self.dataset_dir,
            self.train_dir,
        }

        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)

        num_train_pids = self.get_num_pids(train)

        super(HWData, self).__init__(train, None, None, **kwargs)

    def process_dir(self, dir_path, min_img_size=2, is_train=True, pid_size=None):
        data = []
        pid_folders = os.listdir(dir_path)

        if pid_size:
            pid_folders = pid_folders[:int(pid_size)]
        for img_idx, pid_folder in enumerate(pid_folders):
            img_files = os.listdir(osp.join(dir_path, pid_folder))
            if len(img_files) < min_img_size:
                continue
            else:
                for img_file in img_files:
                    img_path = osp.join(dir_path, pid_folder, img_file)
                    pid =  pid_folder
                    camid = -1
                    if is_train:
                        pid = self.dataset_name + "_" + str(pid)
                        camid = self.dataset_name + "_" + str(camid)
                    data.append((img_path, pid, camid))

        return data
