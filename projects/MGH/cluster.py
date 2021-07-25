import collections
import itertools
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from fastreid.data import build_reid_train_loader
from fastreid.utils import comm
from fastreid.utils.clustering import label_generator_dbscan
from fastreid.utils.metrics import cluster_metrics
from fastreid.utils.torch_utils import extract_features


class Cluster:
    def __init__(self, cfg, model):
        self._logger = logging.getLogger('fastreid.' + __name__)
        self._cfg = cfg
        self.model = model
        self._data_loader_cluster = build_reid_train_loader(self._cfg, is_train=False, for_clustering=True)
        self._common_dataset = self._data_loader_cluster.dataset

        self.num_classes = []
        self.indep_thres = []
        

    def update_labels(self):
        
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
                save_path = '{}/clustering/{}/clustering_epoch{}.pt'.format(self._cfg.OUTPUT_DIR, all_dataset_names[idx], 0)
                start_id, end_id = datasets_size_range[idx], datasets_size_range[idx + 1]
                if os.path.exists(save_path):
                    res = torch.load(save_path)
                    labels, centers, num_classes, indep_thres, dist_mat = res['labels'], res['centers'], res['num_classes'], res['indep_thres'], res['dist_mat']
                else:
                    labels, centers, num_classes, indep_thres, dist_mat = label_generator_dbscan(
                        self._cfg,
                        feats,
                        num_classes=num_classes,
                        indep_thres=indep_thres,
                        epoch=0,
                        cams=cams,
                        imgs_path=img_paths,
                        indexes=indexes
                    )
                if not os.path.exists(save_path):
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    res = {'labels': labels, 'num_classes': num_classes, 'centers': centers, 'indep_thres': indep_thres, 'dist_mat': dist_mat}
                    torch.save(res, save_path)

                if self._cfg.PSEUDO.NORM_CENTER:
                    centers = F.normalize(centers, p=2, dim=1)
            comm.synchronize()

            if comm.is_main_process():
                self.label_summary(labels, true_labels[start_id:end_id], cams, indep_thres=indep_thres, camera_metric=self._cfg.PSEUDO.CAMERA_CLUSTER_METRIC)
            all_labels.append(labels.tolist())
            all_centers.append(centers)
            all_feats.append(feats)
            all_cams.append(cams)

            try:
                self.indep_thres[idx] = indep_thres
            except:
                self.indep_thres.append(indep_thres)
            try:
                self.num_classes[idx] = num_classes
            except:
                self.num_classes.append(num_classes)
        
        return all_labels, all_centers, all_feats, all_cams

    def label_summary(self, pseudo_labels, gt_labels, gt_cameras=np.zeros(100,), cluster_metric=True, indep_thres=None, camera_metric=False):
        if cluster_metric:
            nmi_score, ari_score, purity_score, cluster_acc, precision, recall, fscore, _, _, _ = cluster_metrics(pseudo_labels.long().numpy(), gt_labels.long().numpy(), gt_cameras.long().numpy(), camera_metric)
            self._logger.info(f"nmi_score: {nmi_score*100:.2f}%, ari_score: {ari_score*100:.2f}%, purity_score: {purity_score*100:.2f}%, cluster_acc: {cluster_acc*100:.2f}%, precision: {precision*100:.2f}%, recall: {recall*100:.2f}%, fscore: {fscore*100:.2f}%.")

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

        return clu_num, unclu_ins_num, unused_ins_num