import logging
import numpy as np
import collections

import torch
from sklearn.cluster import DBSCAN

from .faiss_rerank import compute_jaccard_distance
from fastreid.engine.train_loop import HookBase
from fastreid.engine.hooks import *


class ClusterHook(HookBase):
    def __init__(self, eps=0.6, eps_gap=0.02, cluster_iter=200, min_samples=4, metric='precomputed', n_jobs=-1, reset_opt=False):
        self.logger = logging.getLogger('fastreid.' + __name__)
        self.logger.info('Clustering criterion:\t'
                         'eps: {:.3f} eps_gap: {:.3f}\t'
                         'do clustering every {} iterations'.format(eps, eps_gap, cluster_iter))
        self.cluster_iter = cluster_iter
        self.cluster = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=n_jobs)
        self.cluster_tight = DBSCAN(eps=eps - eps_gap, min_samples=min_samples, metric=metric, n_jobs=n_jobs)
        self.cluster_loose = DBSCAN(eps=eps + eps_gap, min_samples=min_samples, metric=metric, n_jobs=n_jobs)
        self.reset_opt = reset_opt

    def before_step(self):
        memory = self.trainer.memory
        if self.trainer.iter % self.cluster_iter == 0:
            features = memory.features.clone()
            rerank_dist = compute_jaccard_distance(features)

            # select & cluster images as training set
            pseudo_labels = self.cluster.fit_predict(rerank_dist)
            pseudo_labels_tight = self.cluster_tight.fit_predict(rerank_dist)
            pseudo_labels_loose = self.cluster_loose.fit_predict(rerank_dist)
            num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
            num_ids_tight = len(set(pseudo_labels_tight)) - (1 if -1 in pseudo_labels_tight else 0)
            num_ids_loose = len(set(pseudo_labels_loose)) - (1 if -1 in pseudo_labels_loose else 0)

            pseudo_labels = self.generate_pseudo_labels(pseudo_labels, num_ids)
            pseudo_labels_tight = self.generate_pseudo_labels(pseudo_labels_tight, num_ids_tight)
            pseudo_labels_loose = self.generate_pseudo_labels(pseudo_labels_loose, num_ids_loose)

            # compute R_indep and R_comp
            N = pseudo_labels.size(0)
            label_sim = pseudo_labels.expand(N, N).eq(pseudo_labels.expand(N, N).t()).float()
            label_sim_tight = pseudo_labels_tight.expand(N, N).eq(pseudo_labels_tight.expand(N, N).t()).float()
            label_sim_loose = pseudo_labels_loose.expand(N, N).eq(pseudo_labels_loose.expand(N, N).t()).float()

            # R_comp and R_indep is 1 - R_comp and 1 - R_indep compared to the description in the paper
            R_comp = 1 - torch.min(label_sim, label_sim_tight).sum(-1) / torch.max(label_sim, label_sim_tight).sum(-1)
            R_indep = 1 - torch.min(label_sim, label_sim_loose).sum(-1) / torch.max(label_sim, label_sim_loose).sum(-1)
            assert ((R_comp.min() >= 0) and (R_comp.max() <= 1))
            assert ((R_indep.min() >= 0) and (R_indep.max() <= 1))

            # compute cluster_R_comp and cluster_R_indep
            cluster_R_comp, cluster_R_indep = collections.defaultdict(list), collections.defaultdict(list)
            cluster_img_num = collections.defaultdict(int)
            for i, (comp, indep, label) in enumerate(zip(R_comp, R_indep, pseudo_labels)):
                cluster_R_comp[label.item()].append(comp.item())
                cluster_R_indep[label.item()].append(indep.item())
                cluster_img_num[label.item()] += 1

            cluster_R_comp = [min(cluster_R_comp[i]) for i in sorted(cluster_R_comp.keys())]
            cluster_R_indep = [min(cluster_R_indep[i]) for i in sorted(cluster_R_indep.keys())]
            cluster_R_indep_noins = [iou for iou, num in zip(cluster_R_indep, sorted(cluster_img_num.keys())) if
                                     cluster_img_num[num] > 1]
            if self.trainer.iter == self.trainer.start_iter:
                self.indep_thres = np.sort(cluster_R_indep_noins)[
                    min(len(cluster_R_indep_noins) - 1, np.round(len(cluster_R_indep_noins) * 0.9).astype('int'))]

            # construct data loader for training
            pseudo_labeled_dataset = []
            outliers = 0
            for i, ((fname, _, cam_id), label) in enumerate(
                    zip(sorted(self.trainer.data_loader.dataset.img_items), pseudo_labels)):
                indep_score = cluster_R_indep[label.item()]
                comp_score = R_comp[i]
                if (indep_score <= self.indep_thres) and (comp_score.item() <= cluster_R_comp[label.item()]):
                    # maintain the cluster id for these reliable instances
                    pseudo_labeled_dataset.append((fname, label.item(), cam_id))
                else:
                    pseudo_labeled_dataset.append((fname, len(cluster_R_indep) + outliers, cam_id))
                    pseudo_labels[i] = len(cluster_R_indep) + outliers
                    outliers += 1

            # statistics of clusters and un-clustered instances
            index2label = collections.defaultdict(int)
            for label in pseudo_labels:
                index2label[label.item()] += 1
            index2label = np.fromiter(index2label.values(), dtype=float)
            self.logger.info('Statistics in iteration {}: {} clusters, {} un-clustered instances, R_indep threshold is {}'
                  .format(self.trainer.iter, (index2label > 1).sum(), (index2label == 1).sum(), 1 - self.indep_thres))

            memory.labels = pseudo_labels.to(self.trainer.cfg.MODEL.DEVICE)
            self.trainer.data_loader = self.trainer.construct_unsupervised_dataloader(pseudo_labeled_dataset,
                                                                                      is_train=True)
            self.trainer._data_loader_iter = iter(self.trainer.data_loader)
            if self.reset_opt:
                self.logger.warning("Reset optimizer at iteration {}".format(self.trainer.iter))
                self.trainer.optimizer.state = collections.defaultdict(dict)

    def generate_pseudo_labels(self, cluster_id, num):
        labels = []
        outliers = 0
        for i, ((fname, _, cam_id), cid) in enumerate(
                zip(sorted(self.trainer.data_loader.dataset.img_items), cluster_id)):
            if cid != -1:
                labels.append(cid)
            else:
                labels.append(num + outliers)
                outliers += 1
        return torch.Tensor(labels).long()
