'''
Author: WuYiming
Date: 2020-10-12 21:22:11
LastEditTime: 2020-10-30 09:44:23
LastEditors: Please set LastEditors
Description: Hooks for SpCL
FilePath: /fast-reid/projects/SpCL_new/hooks.py
'''


import datetime
import collections
import itertools
import logging
from sklearn.metrics import confusion_matrix

import torch
import torch.nn.functional as F
import numpy as np

from fastreid.utils import comm
from fastreid.engine.hooks import *
from fvcore.common.timer import Timer
from fastreid.data import build_reid_train_loader
from fastreid.utils.torch_utils import extract_features
from fastreid.utils.clustering import label_generator_dbscan, label_generator_kmeans
from fastreid.utils.events import get_event_storage


class SALLabelGeneratorHook(LabelGeneratorHook):
    """ Hook for the combination of unsupervised learning and active learning.
    """
    __factory = {
        'dbscan': label_generator_dbscan,
        'kmeans': label_generator_kmeans
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

        # auto-scale for active learning parameters
        if cfg.is_frozen():
            cfg.defrost()
            cfg.ACTIVE.START_ITER *= cfg.DATALOADER.ITERS_PER_EPOCH
            cfg.ACTIVE.TRAIN_ITER *= cfg.DATALOADER.ITERS_PER_EPOCH
            cfg.freeze()
        else:
            cfg.ACTIVE.START_ITER *= cfg.DATALOADER.ITERS_PER_EPOCH
            cfg.ACTIVE.TRAIN_ITER *= cfg.DATALOADER.ITERS_PER_EPOCH

        # Added for active learning
        data_size = self._common_dataset.datasets_size[0]

        # Sampler initialization
        self.sampler = Sampler(cfg)
        self.sampler.query_sample_num = int(cfg.ACTIVE.SAMPLE_M * data_size * cfg.PSEUDO.CLUSTER_ITER// (cfg.SOLVER.MAX_ITER - cfg.ACTIVE.START_ITER))
        self.sampler.data_size = data_size

    def before_step(self):
        if self.trainer.iter % self._cfg.PSEUDO.CLUSTER_ITER == 0 \
                or self.trainer.iter % self._cfg.ACTIVE.TRAIN_ITER == 0 \
                or self.trainer.iter == self.trainer.start_iter:
            self._step_timer.reset()

            # get memory features
            self.get_memory_features()

            # generate pseudo labels and centers
            all_labels, all_centers = self.update_labels()

            # update train loader with pseudo labels
            self.update_train_loader(all_labels)

            # update memory labels
            self.update_memory_labels(all_labels)

            comm.synchronize()

            sec = self._step_timer.seconds()
            self._logger.info(f"Finished updating pseudo label in {str(datetime.timedelta(seconds=int(sec)))}")

    def update_labels(self):
        self._logger.info(f"Start updating pseudo labels on iteration {self.trainer.iter}")
        
        if self.memory_features is None:
            all_features = []
            features, true_labels = extract_features(self.model,
                                                    self._data_loader_cluster,
                                                    self._cfg.PSEUDO.NORM_FEAT)
            all_features.append(features)
            all_features = torch.stack(all_features, dim=0).mean(0)
        else:
            all_features = self.memory_features
            true_labels = torch.LongTensor([self._data_loader_cluster.dataset.pid_dict[item[1]] for item in self._data_loader_cluster.dataset.img_items])

        if self._cfg.PSEUDO.NORM_FEAT:
            all_features = F.normalize(all_features, p=2, dim=1)
        datasets_size = self._common_dataset.datasets_size
        datasets_size_range = list(itertools.accumulate([0] + datasets_size))
        assert len(all_features) == datasets_size_range[-1], f"number of features {len(all_features)} should be same as the unlabeled data size {datasets_size_range[-1]}"

        all_centers = []
        all_labels = []
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
            features = all_features[start_id: end_id]
            gt_labels = true_labels[start_id:end_id]
            if comm.is_main_process():
                # clustering only on first GPU
                labels, centers, num_classes, indep_thres, dist_mat = self.label_generator(
                    self._cfg,
                    features,
                    num_classes=num_classes,
                    indep_thres=indep_thres
                )
                if self._cfg.PSEUDO.NORM_CENTER:
                    centers = F.normalize(centers, p=2, dim=1)
            comm.synchronize()

            # calculate similarity matrix
            # 1. calculate from distance matrix, distance matrix is calculated by jaccard distance, range from 0 to 1, 0 is most different, 1 is most similar.
            weight_matrix = []
            for i in range(num_classes):
                index = torch.where(labels == i)[0].tolist()
                dist_ = dist_mat[:, index].mean(1)
                weight_matrix.append(dist_)
            weight_matrix = 1 - torch.stack(weight_matrix).t()
            # select the score calculated with clustered instances
            # 2. calculate cosine similarity, cosine similarity is more closer than jaccard distance
            # weight_matrix = (features.matmul(centers.t()) + 1) / 2

            if comm.is_main_process():
                clu_num, _, _ = self.label_summary(labels, gt_labels, indep_thres=indep_thres)
                # rectify labels with active learning
                if self._cfg.ACTIVE.RECTIFY:
                    if self.trainer.iter >= self._cfg.ACTIVE.START_ITER and self.sampler.could_sample():
                        clu_sim_mat = weight_matrix[:, :clu_num].clone()
                        self.sampler.sample(clu_sim_mat, dist_mat, labels, gt_labels)
                        labels, centers, num_classes, indep_thres, dist_mat, weight_matrix = self.rectify(features, labels, num_classes, indep_thres, dist_mat)
                        clu_num, _, _ = self.label_summary(labels, gt_labels, indep_thres=indep_thres)
            # use jaccard distance matrix for weight matrix, cosine similarity is too close for different clusters.
            if self._cfg.PSEUDO.MEMORY.WEIGHTED:
                # set weight matrix for calculating weighted contrastive loss.
                self.trainer.weight_matrix = weight_matrix

            # broadcast to other process
            if comm.get_world_size() > 1:
                num_classes = int(comm.broadcast_value(num_classes, 0))
                if self._cfg.PSEUDO.NAME == "dbscan" and len(self._cfg.PSEUDO.DBSCAN.EPS) > 1:
                    # use clustering reliability criterion
                    indep_thres = comm.broadcast_value(indep_thres, 0)
                if comm.get_rank() > 0:
                    labels = torch.arange(len(dataset)).long()
                    centers = torch.zeros((num_classes, all_features.size(-1))).float()
                labels = comm.broadcast_tensor(labels, 0)
                centers = comm.broadcast_tensor(centers, 0)

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

    def rectify(self, features, labels, num_classes, indep_thres, dist_mat, rectify_method='clustering'):
        new_dist_mat = dist_mat.clone()
        # dist_mat is symmetric matrix
        for pos in self.sampler.pos_set:
            new_dist_mat[pos[0], pos[1]] = 0
            new_dist_mat[pos[1], pos[0]] = 0
        for neg in self.sampler.neg_set:
            new_dist_mat[neg[0], neg[1]] = 1
            new_dist_mat[neg[1], neg[0]] = 1

        if rectify_method == 'clustering':
            new_labels, new_centers, new_num_classes, new_indep_thres, new_dist_mat = self.label_generator(
                        self._cfg,
                        features,
                        num_classes=num_classes,
                        indep_thres=indep_thres,
                        dist=new_dist_mat.numpy()
                    )
            if self._cfg.PSEUDO.NORM_CENTER:
                new_centers = F.normalize(new_centers, p=2, dim=1)

            new_weight_matrix = []
            for i in range(new_num_classes):
                index = torch.where(new_labels == i)[0].tolist()
                dist_ = dist_mat[:, index].mean(1)
                new_weight_matrix.append(dist_)
            new_weight_matrix = 1 - torch.stack(new_weight_matrix).t()

        elif rectify_method == 'diffusion':
            raise NotImplemented('Please implement diffusion method.')
        else:
            raise NotImplemented(f'{rectify_method} is not supported in rectifying pseudo labels.')
        
        if not torch.isfinite(new_weight_matrix).all():
            print('Error')
        return new_labels, new_centers, new_num_classes, new_indep_thres, new_dist_mat, new_weight_matrix


class Sampler:
    def __init__(self, cfg):
        self.query_set = set()
        self.pos_set = set()
        self.neg_set = set()
        self.triplet_set = set()
        # sampling limitation
        self.M = cfg.ACTIVE.SAMPLE_M
        self.K = cfg.ACTIVE.SAMPLE_K
        self.top_K = cfg.ACTIVE.PAIR_TOP_RANK
        # sampling method
        self.query_func = cfg.ACTIVE.SAMPLER.QUERY_FUNC
        self.pair_func = cfg.ACTIVE.SAMPLER.PAIR_FUNC
        # extra variables
        self.data_size = 0
        self.query_sample_num = 0
        self._logger = logging.getLogger('fastreid.' + __name__)

    def sample(self, clu_sim_mat, dist_mat, pred_labels, targets):
        # select query
        query_index = self.query_sample(clu_sim_mat)

        # select pair
        selected_pairs = self.pair_sample(query_index, dist_mat, targets)
        self.selected_pairs_summary(selected_pairs, pred_labels, targets)

    def query_sample(self, clu_sim_mat, temp=0.05):
        sim_mat = clu_sim_mat / temp
        assert self.query_sample_num != 0
        assert self.data_size != 0
        if self.query_func == 'sequential_random':
            query_index = self.index_iter.next()
        elif self.query_func == 'random':
            query_index = torch.randperm(self.data_size)[:self.query_sample_num]
        elif self.query_func == 'entropy':
            sim_prob = sim_mat.softmax(dim=1)
            sim_entropy = (-sim_prob * (sim_prob + 1e-6).log()).sum(dim=1)
            query_index = sim_entropy.argsort(descending=True)[:self.query_sample_num]
        elif self.query_func == 'confidence':
            sim_max_prob = sim_mat.max(dim=1)[0]
            query_index = sim_max_prob.argsort()[:self.query_sample_num]
        elif self.query_func == 'diff':
            sim_sorted = sim_mat.sort(descending=True)[0]
            sim_diff = sim_sorted[:, 0] - sim_sorted[:, 1]
            query_index = sim_diff.argsort()[:self.query_sample_num]
        else:
            raise NotImplemented(f"{self.query_func} is not supported in query selection.")
        return query_index

    def pair_sample(self, query_index, dist_mat, targets):
        if self.pair_func == 'random':
            return self.random_sample(query_index, dist_mat, targets)

    def random_sample(self, query_index, dist_mat, targets):
        # for each query, try 5 * self.K times in selecting pairs.
        try_limitation = 5*self.K
        sim_mat = torch.argsort(dist_mat, dim=1)[:, 1:1+max(self.top_K, try_limitation)]
        selected_pairs = []
        for index1 in query_index:
            count = 0
            pos_set = []
            neg_set = []
            index1 = index1.item()
            mask = np.random.permutation(sim_mat.size(1))
            for index2 in sim_mat[index1][mask]:
                if len(pos_set) + len(neg_set) >= self.K or count >= try_limitation:
                    # only select K samples for each query
                    break
                index2 = index2.item()
                if index1 == index2:
                    continue
                selected_pair = tuple(sorted((index1, index2)))
                if selected_pair in self.pos_set or selected_pair in self.neg_set:
                    continue
                selected_pairs.append(selected_pair)
                if targets[index2] == targets[index1]:
                    pos_set.append(index2)
                    self.pos_set.add(selected_pair)
                else:
                    neg_set.append(index2)
                    self.neg_set.add(selected_pair)
                count += 1
            for pos in pos_set:
                for neg in neg_set:
                    self.triplet_set.add((index1, pos, neg))
        return selected_pairs

    def selected_pairs_summary(self, selected_pairs, pred_labels, targets):
        pred_matrix = (pred_labels.view(-1, 1) == pred_labels.view(1, -1))
        targets_matrix = (targets.view(-1, 1) == targets.view(1, -1))
        pred_selected = np.array([pred_matrix[tuple(i)].item() for i in selected_pairs])
        gt_selected = np.array([targets_matrix[tuple(i)].item() for i in selected_pairs])
        tn, fp, fn, tp = confusion_matrix(gt_selected, pred_selected, labels=[0,1]).ravel()  # labels=[0,1] to force output 4 values
        self._logger.info('Active selector summary: ')
        self._logger.info('            |       |     Prediction     |')
        self._logger.info('------------------------------------------')
        self._logger.info('            |       |  True    |  False  |')
        self._logger.info('------------------------------------------')
        self._logger.info('GroundTruth | True  |   {:5d}  |  {:5d}  |'.format(tp, fn))
        self._logger.info('            | False |   {:5d}  |  {:5d}  |'.format(fp, tn))
        self._logger.info(f'size of pos_set is {len(self.pos_set)}')
        self._logger.info(f'size of neg_set is {len(self.neg_set)}')
        self._logger.info(f'size of triplet_set is {len(self.triplet_set)}')
        # storage summary into tensorboard
        storage = get_event_storage()
        storage.put_scalar("tn", tn)
        storage.put_scalar("fp", fp)
        storage.put_scalar("fn", fn)
        storage.put_scalar("tp", tp)

    def could_sample(self):
        return (len(self.pos_set.union(self.neg_set))) < self.K * self.data_size or len(self.sampler.query_set) < self.M * self.data_size