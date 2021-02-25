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
from fastreid.data import build_reid_train_loader
from fastreid.utils.torch_utils import extract_features
from fastreid.engine.hooks import LabelGeneratorHook


class CAPLabelGeneratorHook(LabelGeneratorHook):
    def before_epoch(self):
        if self.trainer.epoch % self._cfg.PSEUDO.CLUSTER_EPOCH == 0 \
                or self.trainer.epoch == self.trainer.start_epoch:
            self._step_timer.reset()

            # get memory features
            self.memory_features = None

            # generate pseudo labels and centers
            all_labels, all_centers, all_features, all_camids, all_pseudo_centers = self.update_labels()

            # update train loader
            self.update_train_loader(all_labels)

            # reset optimizer
            if self._cfg.PSEUDO.RESET_OPT:
                self._logger.info(f"Reset optimizer")
                self.trainer.optimizer.state = collections.defaultdict(dict)

            self.trainer.memory._update_center(all_centers[0], all_labels[0], all_camids[0])
            self.trainer.memory._update_epoch(self.trainer.epoch)

            # update classifier centers
            if self._cfg.PSEUDO.WITH_CLASSIFIER:
                self.update_classifier_centers(all_pseudo_centers)

            comm.synchronize()

            sec = self._step_timer.seconds()
            self._logger.info(f"Finished updating pseudo label in {str(datetime.timedelta(seconds=int(sec)))}")

    def update_labels(self):
        self._logger.info(f"Start updating pseudo labels on epoch {self.trainer.epoch}/iteration {self.trainer.iter}")
        
        all_features, true_labels, img_paths, all_camids, indexes = extract_features(self.model,
                                                                    self._data_loader_cluster,
                                                                    self._cfg.PSEUDO.NORM_FEAT)

        if self._cfg.PSEUDO.NORM_FEAT:
            all_features = F.normalize(all_features, p=2, dim=1)
        datasets_size = self._common_dataset.datasets_size
        datasets_size_range = list(itertools.accumulate([0] + datasets_size))
        assert len(all_features) == datasets_size_range[-1], f"number of features {len(all_features)} should be same as the unlabeled data size {datasets_size_range[-1]}"

        all_centers = []
        all_labels = []
        all_feats = []
        all_cams = []
        all_pseudo_centers = []
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

            start_id, end_id = datasets_size_range[idx], datasets_size_range[idx + 1]
            feats = all_features[start_id: end_id]
            cams = all_camids[start_id: end_id]
            if comm.is_main_process():
                # clustering only on first GPU
                save_path = '{}/clustering/{}/clustering_epoch{}.pt'.format(self._cfg.OUTPUT_DIR, all_dataset_names[idx], self.trainer.epoch)
                if self._cfg.PSEUDO.TRUE_LABEL:
                    labels = true_labels[start_id: end_id]
                    num_classes = len(set(labels))
                    pseudo_centers = collections.defaultdict(list)
                    for i, label in enumerate(labels):
                        pseudo_centers[labels[i].item()].append(feats[i])

                    pseudo_centers = [torch.stack(pseudo_centers[idx], dim=0).mean(0) for idx in sorted(pseudo_centers.keys())]
                    pseudo_centers = torch.stack(pseudo_centers, dim=0)
                else:
                    if os.path.exists(save_path):
                        res = torch.load(save_path)
                        labels, pseudo_centers, num_classes, indep_thres, dist_mat = res['labels'], res['pseudo_centers'], res['num_classes'], res['indep_thres'], res['dist_mat']
                    else:
                        labels, pseudo_centers, num_classes, indep_thres, dist_mat = self.label_generator(
                            self._cfg,
                            feats,
                            num_classes=num_classes,
                            indep_thres=indep_thres,
                            epoch=self.trainer.epoch,
                            cams=cams,
                            imgs_path=img_paths,
                            indexes=indexes
                        )
            comm.synchronize()

            # broadcast to other process
            if comm.get_world_size() > 1:
                num_classes = int(comm.broadcast_value(num_classes, 0))
                if self._cfg.PSEUDO.NAME == "dbscan" and len(self._cfg.PSEUDO.DBSCAN.EPS) > 1:
                    # use clustering reliability criterion
                    indep_thres = comm.broadcast_value(indep_thres, 0)
                if comm.get_rank() > 0:
                    labels = torch.arange(len(dataset)).long()
                    pseudo_centers = torch.zeros((num_classes, feats.size(-1))).float()
                    if self._cfg.PSEUDO.MEMORY.WEIGHTED:
                        dist_mat = torch.zeros((len(dataset), len(dataset))).float()
                labels = comm.broadcast_tensor(labels, 0)
                pseudo_centers = comm.broadcast_tensor(pseudo_centers, 0)
                if self._cfg.PSEUDO.MEMORY.WEIGHTED:
                    dist_mat = comm.broadcast_tensor(dist_mat, 0)

            if comm.is_main_process():
                if not os.path.exists(save_path) and self._cfg.PSEUDO.SAVE_CLUSTERING_RES:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    res = {'labels': labels, 'num_classes': num_classes, 'indep_thres': indep_thres, 'dist_mat': dist_mat, 'pseudo_centers': pseudo_centers}
                    torch.save(res, save_path)
                self.label_summary(labels, true_labels[start_id:end_id], cams, indep_thres=indep_thres, camera_metric=self._cfg.PSEUDO.CAMERA_CLUSTER_METRIC)

            # camera-aware proxies/centers
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1: continue  # remove the ourliers
                centers[(label.item(), cams[i].item())].append(feats[i])
            centers = {k: torch.stack(centers[k], dim=0).mean(0, keepdim=True) for k in centers}
            
            if self._cfg.PSEUDO.NORM_CENTER:
                centers = {k: F.normalize(centers[k], p=2, dim=1) for k in centers}
                pseudo_centers = F.normalize(pseudo_centers, p=2, dim=1)


            all_labels.append(labels.tolist())
            all_centers.append(centers)
            all_feats.append(feats)
            all_cams.append(cams)
            all_pseudo_centers.append(pseudo_centers)

            try:
                self.indep_thres[idx] = indep_thres
            except:
                self.indep_thres.append(indep_thres)
            try:
                self.num_classes[idx] = num_classes
            except:
                self.num_classes.append(num_classes)

        return all_labels, all_centers, all_feats, all_cams, all_pseudo_centers

