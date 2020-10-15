'''
Author: WuYiming
Date: 2020-10-12 21:22:11
LastEditTime: 2020-10-14 16:06:27
LastEditors: Please set LastEditors
Description: Hooks for SpCL
FilePath: /fast-reid/projects/SpCL_new/hooks.py
'''

import datetime
import collections

import torch

from fastreid.utils import comm
from fastreid.engine.hooks import *


class SpCLLabelGeneratorHook(LabelGeneratorHook):

    def before_step(self):
        if self.trainer.iter % self._cfg.PSEUDO.CLUSTER_ITER == 0 \
                or self.trainer.iter == self.trainer.start_iter:
            self._step_timer.reset()

            # get memory features
            self.get_memory_features()

            # generate pseudo labels and centers
            all_labels, all_centers = self.update_labels()

            # update train loader
            self.update_train_loader(all_labels)

            # update memory labels
            self.update_memory_labels(all_labels)

            comm.synchronize()

            sec = self._step_timer.seconds()
            self._logger.info(f"Finished updating pseudo label in {str(datetime.timedelta(seconds=int(sec)))}")

    def update_memory_labels(self, all_labels):
        sup_commdataset = self._data_loader_sup.dataset
        sup_datasets = sup_commdataset.datasets
        memory_labels = []
        start_pid = 0
        for idx, dataset in enumerate(sup_datasets):
            if idx in self._cfg.PSEUDO.UNSUP:
                labels = all_labels[self._cfg.PSEUDO.UNSUP.index(idx)]
                memory_labels.append(torch.LongTensor(labels) + start_pid)
                start_pid += max(labels) + 1
            else:
                num_pids = dataset.num_train_pids
                memory_labels.append(torch.arange(start_pid, start_pid + num_pids))
                start_pid += num_pids
        memory_labels = torch.cat(memory_labels).view(-1)
        self.trainer.memory._update_label(memory_labels)