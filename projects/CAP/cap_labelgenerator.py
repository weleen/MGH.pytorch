import datetime
import itertools
import logging
import os
import tempfile
import time
from collections import Counter
import copy
import numpy as np
import collections

import torch
import torch.nn.functional as F
from torch import nn

from fastreid.utils import comm
from fvcore.common.timer import Timer
from fastreid.data import build_reid_train_loader
from fastreid.utils.clustering import label_generator_dbscan, label_generator_kmeans, label_generator_cdp
from fastreid.utils.metrics import cluster_metrics
from fastreid.utils.torch_utils import extract_features
from fastreid.engine.train_loop import HookBase


class CAPLabelGeneratorHook(HookBase):
    """Generate pseudo labels 
    """
    __factory = {
        'dbscan': label_generator_dbscan,
        'kmeans': label_generator_kmeans,
        'cdp': label_generator_cdp
    }

    def __init__(self, cfg, model):
        self._logger = logging.getLogger('fastreid.' + __name__)
        self._step_timer = Timer()
        self._cfg = cfg
        self.model = model
        # only build the data loader for unlabeled dataset
        self._logger.info("Build the dataloader for clustering.")
        self._data_loader_cluster = build_reid_train_loader(cfg, is_train=False, for_clustering=True)
        # save the original unlabeled dataset info
        self._common_dataset = self._data_loader_cluster.dataset

        assert cfg.PSEUDO.ENABLED, "pseudo label settings are not enabled."
        assert cfg.PSEUDO.NAME in self.__factory.keys(), \
            f"{cfg.PSEUDO.NAME} is not supported, please select from {self.__factory.keys()}"
        self.label_generator = self.__factory[cfg.PSEUDO.NAME]

        self.num_classes = []
        self.indep_thres = []
        if cfg.PSEUDO.NAME == 'kmeans':
            self.num_classes = cfg.PSEUDO.NUM_CLUSTER

    def before_train(self):
        # save the original all dataset info
        self._logger.info("Copy the dataloader from original dataloader with all groundtruth information.")
        self._data_loader_sup = copy.deepcopy(self.trainer.data_loader)

    def before_epoch(self):
        if self.trainer.epoch % self._cfg.PSEUDO.CLUSTER_EPOCH == 0 \
                or self.trainer.epoch == self.trainer.start_epoch:
            self._step_timer.reset()

            # get memory features
            self.memory_features = None

            # generate pseudo labels and centers
            all_labels, all_centers = self.update_labels()

            # update train loader
            self.update_train_loader(all_labels)

            # reset optimizer
            if self._cfg.PSEUDO.RESET_OPT:
                self._logger.info(f"Reset optimizer")
                self.trainer.optimizer.state = collections.defaultdict(dict)

            self.trainer.memory._update_center(all_centers[0])

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

    def update_labels(self):
        self._logger.info(f"Start updating pseudo labels on epoch {self.trainer.epoch}/iteration {self.trainer.iter}")
        
        all_features, true_labels, _ = extract_features(self.model,
                                                        self._data_loader_cluster,
                                                        self._cfg.PSEUDO.NORM_FEAT)

        all_camids = self.get_camids(self._data_loader_cluster)

        if self._cfg.PSEUDO.NORM_FEAT:
            all_features = F.normalize(all_features, p=2, dim=1)
        datasets_size = self._common_dataset.datasets_size
        datasets_size_range = list(itertools.accumulate([0] + datasets_size))
        assert len(all_features) == datasets_size_range[-1], f"number of features {len(all_features)} should be same as the unlabeled data size {datasets_size_range[-1]}"

        all_centers = []
        all_labels = []
        all_dataset_names = [self._cfg.DATASETS.NAMES[ind] for ind in self._cfg.PSEUDO.UNSUP]
        for idx, dataset in enumerate(self._common_dataset.datasets):

            try:
                indep_thres = self.indep_thres[idx]
            except:
                indep_thres = None
            try:
                num_classes = self.num_classes[idx]
            except:
                num_classes = None

            if comm.is_main_process():
                # clustering only on first GPU
                save_path = '{}/clustering/{}/clustering_epoch{}.pt'.format(self._cfg.OUTPUT_DIR, all_dataset_names[idx], self.trainer.epoch)
                start_id, end_id = datasets_size_range[idx], datasets_size_range[idx + 1]
                # if os.path.exists(save_path):
                #     res = torch.load(save_path)
                #     labels, centers, num_classes, indep_thres, dist_mat = res['labels'], res['centers'], res['num_classes'], res['indep_thres'], res['dist_mat']
                # else:
                labels, centers, num_classes, indep_thres, dist_mat = self.label_generator(
                    self._cfg,
                    all_features[start_id: end_id],
                    num_classes=num_classes,
                    indep_thres=indep_thres,
                    epoch=self.trainer.epoch
                )

                # camera-aware proxies/centers
                features = all_features[start_id: end_id]
                camids = all_camids[start_id: end_id]
                centers = collections.defaultdict(list)
                for i, label in enumerate(labels):
                    centers[(label.item(), camids[i].item())].append(features[i])

                # remove outliers
                centers = {k: torch.stack(centers[k], dim=0).mean(0, keepdim=True) for k in centers if len(centers[k]) > 1}

                if not os.path.exists(save_path):
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    res = {'labels': labels, 'num_classes': num_classes}#, 'centers': centers, 'indep_thres': indep_thres, 'dist_mat': dist_mat}
                    torch.save(res, save_path)

                if self._cfg.PSEUDO.NORM_CENTER:
                    centers = {k: F.normalize(centers[k], p=2, dim=1) for k in centers}
            comm.synchronize()

            # broadcast to other process
            # if comm.get_world_size() > 1:
            #     num_classes = int(comm.broadcast_value(num_classes, 0))
            #     if self._cfg.PSEUDO.NAME == "dbscan" and len(self._cfg.PSEUDO.DBSCAN.EPS) > 1:
            #         # use clustering reliability criterion
            #         indep_thres = comm.broadcast_value(indep_thres, 0)
            #     if comm.get_rank() > 0:
            #         labels = torch.arange(len(dataset)).long()
            #         centers = torch.zeros((num_classes, all_features.size(-1))).float()
            #         if self._cfg.PSEUDO.MEMORY.WEIGHTED:
            #             dist_mat = torch.zeros((len(dataset), len(dataset))).float()
            #     labels = comm.broadcast_tensor(labels, 0)
            #     centers = comm.broadcast_tensor(centers, 0)
            #     if self._cfg.PSEUDO.MEMORY.WEIGHTED:
            #         dist_mat = comm.broadcast_tensor(dist_mat, 0)

            # calculate weight_matrix if use weight_matrix in loss calculation
            if self._cfg.PSEUDO.MEMORY.WEIGHTED:
                assert len(self._cfg.DATASETS.NAMES) == 1, 'Only single single dataset is supported for calculating weight_matrix'
                weight_matrix = torch.zeros((len(labels), num_classes))
                nums = torch.zeros((1, num_classes))
                index_select = torch.where(labels >= 0)[0]
                inputs_select = dist_mat[index_select]
                labels_select = labels[index_select]
                weight_matrix.index_add_(1, labels_select, inputs_select)
                nums.index_add_(1, labels_select, torch.ones(1, len(index_select)))
                weight_matrix = 1 - weight_matrix / nums
                self.trainer.weight_matrix = weight_matrix

            if comm.is_main_process():
                self.label_summary(labels, true_labels[start_id:end_id], indep_thres=indep_thres)
            all_labels.append(labels.tolist())
            all_centers.append(centers)

            try:
                self.indep_thres[idx] = indep_thres
            except:
                self.indep_thres.append(indep_thres)
            try:
                self.num_classes[idx] = num_classes
            except:
                self.num_classes.append(num_classes)

        return all_labels, all_centers

    def get_camids(self, data_loader):
        total = len(data_loader)
        data_iter = iter(data_loader)
        camids = list()
        for idx in range(total):
            inputs = next(data_iter)
            camids.append(inputs["camids"])
        camids = torch.cat(camids, dim=0)

        return camids

    def update_train_loader(self, all_labels):
        # Here is tricky, we take the datasets from self._data_loader_sup, this datasets is created same as supervised learning.
        sup_commdataset = self._data_loader_sup.dataset
        sup_datasets = sup_commdataset.datasets
        # add the ground truth labels into the all_labels
        pid_labels = list()
        start_pid = 0

        for idx, dataset in enumerate(sup_datasets):
            # get ground truth pid labels start from 0
            pid_lbls = [pid for _, pid, _ in dataset.data]
            min_pid = min(pid_lbls)
            pid_lbls = [pid_lbl - min_pid for pid_lbl in pid_lbls]
            
            if idx in self._cfg.PSEUDO.UNSUP:
                # replace gt labels with pseudo labels
                unsup_idx = self._cfg.PSEUDO.UNSUP.index(idx)
                pid_lbls = all_labels[unsup_idx]

            pid_lbls = [pid + start_pid if pid != -1 else pid for pid in pid_lbls]
            start_pid = max(pid_lbls) + 1
            pid_labels.append(pid_lbls)            

        self.trainer.data_loader = build_reid_train_loader(self._cfg,
                                                           datasets=copy.deepcopy(sup_datasets),  # copy the sup_datasets
                                                           pseudo_labels=pid_labels,
                                                           is_train=True,
                                                           relabel=False)
        self.trainer._data_loader_iter = iter(self.trainer.data_loader)
        # update cfg
        if self._cfg.is_frozen():
            self._cfg.defrost()
            self._cfg.MODEL.HEADS.NUM_CLASSES = self.trainer.data_loader.dataset.num_classes
            self._cfg.freeze()
        else:
            self._cfg.MODEL.HEADS.NUM_CLASSES = self.trainer.data_loader.dataset.num_classes

    def label_summary(self, pseudo_labels, gt_labels, cluster_metric=True, indep_thres=None):
        if cluster_metric:
            nmi_score, ari_score, purity_score, cluster_accuracy = cluster_metrics(pseudo_labels.long().numpy(), gt_labels.long().numpy())
            self._logger.info(f"nmi_score: {nmi_score*100:.2f}%, ari_score: {ari_score*100:.2f}%, purity_score: {purity_score*100:.2f}%")
            self.trainer.storage.put_scalar('nmi_score', nmi_score, smoothing_hint=False)
            self.trainer.storage.put_scalar('ari_score', ari_score, smoothing_hint=False)
            self.trainer.storage.put_scalar('purity_score', purity_score, smoothing_hint=False)
            self.trainer.storage.put_scalar('cluster_accuracy', cluster_accuracy, smoothing_hint=False)

        # statistics of clusters and un-clustered instances
        index2label = collections.defaultdict(int)
        for label in pseudo_labels.tolist():
            index2label[label] += 1
        unused_ins_num = index2label.pop(-1) if -1 in index2label.keys() else 0
        index2label = np.array(list(index2label.values()))
        clu_num = (index2label > 1).sum()
        unclu_ins_num = (index2label == 1).sum()

        if indep_thres is None:
            self._logger.info(f"Cluster Statistic: clusters {clu_num}, un-clustered instances {unclu_ins_num}, unused instances {unused_ins_num}")
        else:
            self._logger.info(f"Cluster Statistic: clusters {clu_num}, un-clustered instances {unclu_ins_num}, unused instances {unused_ins_num}, R_indep threshold is {1 - indep_thres}")

        self.trainer.storage.put_scalar('num_clusters', clu_num, smoothing_hint=False)
        self.trainer.storage.put_scalar('num_outliers', unclu_ins_num, smoothing_hint=False)
        self.trainer.storage.put_scalar('num_unused_instances', unused_ins_num, smoothing_hint=False)

        return clu_num, unclu_ins_num, unused_ins_num