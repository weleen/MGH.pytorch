'''
Author: WuYiming
Date: 2020-10-12 21:22:11
LastEditTime: 2020-11-18 11:10:20
LastEditors: Please set LastEditors
Description: Hooks for SpCL
FilePath: /fast-reid/projects/SpCL_new/hooks.py
'''

import time
import datetime
import collections
import itertools
import logging
from sklearn.metrics import confusion_matrix
import faiss
import torch
import torch.nn.functional as F
import numpy as np
import scipy

from fastreid.utils.metrics import compute_distance_matrix
from fastreid.utils import comm
from fastreid.engine.hooks import *
from fvcore.common.timer import Timer
from fastreid.data import build_reid_train_loader
from fastreid.utils.torch_utils import extract_features
from fastreid.utils.clustering import label_generator_dbscan, label_generator_kmeans, label_generator_cdp
from sampling_method.sampler import dist2classweight, set_labeled_instances, InstaceSampler, PairedSampler, TripletSampler


class SALLabelGeneratorHook(LabelGeneratorHook):
    """ Hook for the combination of unsupervised learning and active learning.
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
        assert len(self._common_dataset.datasets) == 1, "Only support single dataset."

        assert cfg.PSEUDO.ENABLED, "pseudo label settings are not enabled."
        assert cfg.PSEUDO.NAME in self.__factory.keys(), f"{cfg.PSEUDO.NAME} is not supported, please select from {self.__factory.keys()}"
        self.label_generator = self.__factory[cfg.PSEUDO.NAME]

        self.num_classes = None
        self.indep_thres = None
        if cfg.PSEUDO.NAME == 'kmeans':
            self.num_classes = cfg.PSEUDO.NUM_CLUSTER

        if cfg.ACTIVE.SAMPLING_METHOD == 'instance':
            self.sampler = InstaceSampler(cfg, dataset=self._common_dataset.img_items)
        elif cfg.ACTIVE.SAMPLING_METHOD == 'pair':
            self.sampler = PairedSampler(cfg, dataset=self._common_dataset.img_items)
        elif cfg.ACTIVE.SAMPLING_METHOD == 'triplet':
            self.sampler = TripletSampler(cfg, dataset=self._common_dataset.img_items)
        else:
            raise ValueError("Only support sampling method from [instance | pair | triplet], {} is not recognized.".format(cfg.ACTIVE.SAMPLING_METHOD))

    def could_active_sampling(self):
        return self.trainer.epoch % self._cfg.ACTIVE.SAMPLE_EPOCH == 0 \
                and self.trainer.epoch >= self._cfg.ACTIVE.START_EPOCH \
                and self.trainer.epoch < self._cfg.ACTIVE.END_EPOCH \
                and (self._cfg.ACTIVE.RECTIFY or self._cfg.ACTIVE.BUILD_DATALOADER)

    def could_pseudo_labeling(self):
        return self.trainer.epoch % self._cfg.PSEUDO.CLUSTER_EPOCH == 0 \
            or self.trainer.iter == self.trainer.start_iter

    def before_epoch(self):
        if self.could_pseudo_labeling() or self.could_active_sampling():
            self._step_timer.reset()

            # get memory features
            self.get_memory_features()

            # generate pseudo labels and centers
            all_labels, all_centers, active_labels = self.update_labels()

            self._logger.info(f"Update unsupervised data loader.")
            self.update_train_loader(all_labels)
            # update memory labels
            self.update_memory_labels(all_labels)

            if self.could_active_sampling() and self._cfg.ACTIVE.BUILD_DATALOADER:
                self._logger.info(f"Update active data loader.")
                self.update_active_loader(active_labels)

            comm.synchronize()

            sec = self._step_timer.seconds()
            self._logger.info(f"Finished updating label in {str(datetime.timedelta(seconds=int(sec)))}")

    def update_labels(self):
        self._logger.info(f"Start updating pseudo labels on epoch {self.trainer.epoch}/iteration {self.trainer.iter}")
        
        if self.memory_features is None:
            features, gt_labels = extract_features(self.model,
                                                     self._data_loader_cluster,
                                                     self._cfg.PSEUDO.NORM_FEAT)
        else:
            features = self.memory_features
            gt_labels = torch.LongTensor([self._data_loader_cluster.dataset.pid_dict[item[1]] for item in self._data_loader_cluster.dataset.img_items])

        if self._cfg.PSEUDO.NORM_FEAT:
            features = F.normalize(features, p=2, dim=1)

        all_centers = []
        all_labels = []
        indep_thres = self.indep_thres
        num_classes = self.num_classes

        # clustering only on first GPU, this step is time-consuming
        labels, centers, num_classes, indep_thres, dist_mat = self.label_generator(
            self._cfg,
            features,
            num_classes=num_classes,
            indep_thres=indep_thres
        )
        if self._cfg.PSEUDO.NORM_CENTER:
            centers = F.normalize(centers, p=2, dim=1)
        comm.synchronize()

        # calculate class-wise weight matrix
        class_weight_matrix = dist2classweight(dist_mat, num_classes, labels)
        cluster_num, _, _ = self.label_summary(labels, gt_labels, indep_thres=indep_thres)

        # active sampling 
        if self.could_active_sampling() and self.sampler.could_sample():
            self.sampler.query_sample(dist_mat, labels, num_classes, gt_labels, cluster_num=cluster_num, features=features, centers=centers)

        # node label propagation
        if self._cfg.ACTIVE.NODE_PROP and self.sampler.index_label.sum() > 0:
            active_labels = self.node_label_propagation(features.cpu().numpy(), gt_labels)
            classes = list(set(active_labels))
            active_labels = [classes.index(i) for i in active_labels]
        else:
            labeled_index = list(self.sampler.query_set)
            gt_query = gt_labels[labeled_index].tolist()
            classes = list(set(gt_query))
            active_labels = [classes.index(gt_query[labeled_index.index(i)]) if i in labeled_index else -1 for i in range(len(labels))]

        # edge label propagation and rectification
        if self._cfg.ACTIVE.RECTIFY and self.sampler.index_label.sum() > 0:
            labels, centers, num_classes, indep_thres, dist_mat, class_weight_matrix = self.rectify(features, labels, num_classes, indep_thres, dist_mat, gt_labels)
        
        if self._cfg.PSEUDO.MEMORY.WEIGHTED:
            # set weight matrix for calculating weighted contrastive loss.
            self.trainer.weight_matrix = class_weight_matrix

        if self._cfg.ACTIVE.NODE_PROP and self.sampler.index_label.sum() > 0:
            all_labels.append(active_labels)
        else:
            all_labels.append(labels.tolist())
        all_centers.append(centers)

        self.indep_thres = indep_thres
        self.num_classes = num_classes

        return all_labels, all_centers, active_labels

    def rectify(self, features, labels, num_classes, indep_thres, dist_mat, gt_labels):
        if self._cfg.ACTIVE.EDGE_PROP:
            new_dist_mat = self.edge_label_propagation(dist_mat, gt_labels, max_step=self._cfg.ACTIVE.EDGE_PROP_STEP)
            torch.cuda.empty_cache()
        else:
            new_dist_mat, labeled_dist_mat = set_labeled_instances(self.sampler, dist_mat, gt_labels)

        new_labels, new_centers, new_num_classes, new_indep_thres, new_dist_mat = self.label_generator(
                    self._cfg,
                    features,
                    num_classes=num_classes,
                    indep_thres=indep_thres,
                    dist=new_dist_mat.numpy()
                )
        if self._cfg.PSEUDO.NORM_CENTER:
            new_centers = F.normalize(new_centers, p=2, dim=1)

        new_weight_matrix = dist2classweight(new_dist_mat, new_num_classes, new_labels)
        _, _, _ = self.label_summary(new_labels, gt_labels, indep_thres=new_indep_thres)
        return new_labels, new_centers, new_num_classes, new_indep_thres, new_dist_mat, new_weight_matrix

    def update_active_loader(self, pseudo_labels, **kwargs):
        self._logger.info('Build active dataloader.')

        sup_commdataset = self.trainer.data_loader.dataset
        sup_datasets = sup_commdataset.datasets

        re_gt_set = pseudo_labels
        dataset = sup_datasets[0].data
        cam_labels = [dataset[i][2] for i in range(len(dataset))]
        
        gt_dict = dict()
        i = 0
        for p in re_gt_set:
            if gt_dict.get(p) is None:
                gt_dict[p] = i
                i += 1
        re_gt_set = [gt_dict[p] for p in re_gt_set]

        self.trainer.data_loader_active = build_reid_train_loader(self._cfg,
                                                        datasets=sup_datasets,
                                                        pseudo_labels=[re_gt_set],
                                                        cam_labels=[cam_labels],
                                                        is_train=True,
                                                        relabel=False)
        self.trainer._data_loader_active_iter = iter(self.trainer.data_loader_active)
        # update cfg
        if self._cfg.is_frozen():
            self._cfg.defrost()
            self._cfg.MODEL.HEADS.NUM_CLASSES = self.trainer.data_loader_active.dataset.num_classes
            self._cfg.freeze()
        else:
            self._cfg.MODEL.HEADS.NUM_CLASSES = self.trainer.data_loader_active.dataset.num_classes

    def edge_label_propagation(self, dist_mat: torch.Tensor, gt_labels, step=50, alpha=0.99, debug=False):
        # TODO: try cdp graph propagation
        """
        edge label propagation. 
        Ensemble Diffusion for Retrieval, eq(3)
        A(t+1) = αS A(t) S.t() + (1 − α)I, S = D^(−1/2) W D^(−1/2), in our code W = A
        """
        start_time = time.time()
        # mask for residual regression
        dist_mat = dist_mat.cuda('cuda:1')
        corrected_dist_mat, labeled_dist_mat = set_labeled_instances(self.sampler, dist_mat, gt_labels)
        mask = (labeled_dist_mat > -1).float()
        error_dist_mat = corrected_dist_mat - dist_mat

        N = dist_mat.size(0)
        I = torch.eye(N).to(dist_mat.device)
        adj = 1 - dist_mat - I
        D = adj.sum(dim=1).pow(-0.5).view(-1, 1)
        D = D.matmul(D.t())
        S = adj * D

        im_time = time.time()
        if debug: self._logger.info('preprocess cost {} s'.format(im_time - start_time))
        # iterative propagation
        for i in range(step):
            time1 = time.time()
            error_dist_mat = (1 - mask) * (alpha * (S @ error_dist_mat @ S) + (1 - alpha) * I) + mask * labeled_dist_mat
            time2 = time.time()
            if debug: self._logger.info('iteration {} cost {} s'.format(i, time2 - time1))
        time3 = time.time()
        new_dist_mat = (error_dist_mat + dist_mat).clamp(0, 1)
        new_dist_mat = (1 - mask) * new_dist_mat + mask * labeled_dist_mat
        del S, adj, D, error_dist_mat, labeled_dist_mat
        if debug: self._logger.info('residual label propagation cost {} s'.format(time.time() - time3))
        return new_dist_mat.cpu().data

    def node_label_propagation(self, X, true_labels, k = 50, max_step = 20) -> list: 
        self._logger.info('Perform node label propagation')
        alpha = 0.99
        labels = np.asarray(true_labels)
        query_set = self.sampler.query_set
        labeled_idx = np.asarray(list(query_set))
        gt_set = true_labels[list(query_set)].tolist()
        classes = np.unique(gt_set)

        # kNN search for the graph
        d = X.shape[1]
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = int(torch.cuda.device_count()) - 1
        index = faiss.GpuIndexFlatIP(res,d,flat_config)   # build the index

        faiss.normalize_L2(X)
        index.add(X) 
        N = X.shape[0]
        Nidx = index.ntotal

        c = time.time()
        D, I = index.search(X, k + 1)
        elapsed = time.time() - c
        self._logger.info('kNN Search done in %d seconds' % elapsed)

        # only k nearest propagation
        mask = I[labeled_idx][:, :self._cfg.ACTIVE.NODE_PROP_STEP + 1].reshape(-1)
        
        # Create the graph
        D = D[:,1:] ** 3
        I = I[:,1:]
        row_idx = np.arange(N)
        row_idx_rep = np.tile(row_idx,(k,1)).T
        W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
        W = W + W.T

        # Normalize the graph
        W = W - scipy.sparse.diags(W.diagonal())
        S = W.sum(axis = 1)
        S[S==0] = 1
        D = np.array(1./ np.sqrt(S))
        D = scipy.sparse.diags(D.reshape(-1))
        Wn = D * W * D

        # Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply label propagation
        Z = np.zeros((N, len(classes)))
        A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn
        for i in range(len(classes)):
            cur_idx = labeled_idx[np.where(labels[labeled_idx] == classes[i])]
            y = np.zeros((N,))
            y[cur_idx] = 1.0 / cur_idx.shape[0]
            f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_step)
            Z[:,i] = f

        # Handle numberical errors
        Z[Z < 0] = 0 

        # Compute the weight for each instance based on the entropy (eq 11 from the paper)
        probs_l1 = F.normalize(torch.tensor(Z),1).numpy()
        probs_l1[probs_l1 <0] = 0
        entropy = scipy.stats.entropy(probs_l1.T)
        weights = 1 - entropy / np.log(len(classes))
        weights = weights / np.max(weights)
        p_labels = np.argmax(probs_l1,1)

        masked_labels = p_labels.copy()
        if self._cfg.ACTIVE.NODE_PROP_STEP != -1: 
            masked_labels[list(set(range(N))-set(mask))] = -1
        active_labels = masked_labels.tolist()

        p_labels = np.array([classes[i] for i in p_labels])
        # Compute the accuracy of pseudo labels for statistical purposes
        correct_idx = (p_labels == labels)
        acc = correct_idx.mean()
        self._logger.info(f"label propagation acc: %.2f" % (acc * 100))

        masked_labels = np.array([classes[i] if i != -1 else -1 for i in masked_labels])
        # Compute the accuracy of pseudo labels for statistical purposes
        correct_idx = (masked_labels == labels)
        acc = correct_idx.mean()
        self._logger.info(f"masked label propagation acc: %.2f" % (acc * 100))
 
        return active_labels

