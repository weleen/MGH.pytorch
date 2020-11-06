'''
Author: WuYiming
Date: 2020-10-12 21:22:11
LastEditTime: 2020-11-06 15:36:37
LastEditors: Please set LastEditors
Description: Hooks for SpCL
FilePath: /fast-reid/projects/SpCL_new/hooks.py
'''

import time
import datetime
import collections
from fastreid.utils.metrics import compute_distance_matrix
from fastreid.engine.train_loop import HookBase
import itertools
import logging
from sklearn.metrics import confusion_matrix
import faiss
import torch
import torch.nn.functional as F
import numpy as np
import scipy
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
        assert len(self._common_dataset.datasets) == 1, "Support only single dataset."
        # unsupervised learning
        assert cfg.PSEUDO.ENABLED, "pseudo label settings are not enabled."
        assert cfg.PSEUDO.NAME in self.__factory.keys(), \
            f"{cfg.PSEUDO.NAME} is not supported, please select from {self.__factory.keys()}"
        self.label_generator = self.__factory[cfg.PSEUDO.NAME]

        self.num_classes = None
        self.indep_thres = None
        if cfg.PSEUDO.NAME == 'kmeans':
            self.num_classes = cfg.PSEUDO.NUM_CLUSTER

        # active learning
        if cfg.is_frozen():
            cfg.defrost()
            cfg.ACTIVE.START_ITER *= cfg.DATALOADER.ITERS_PER_EPOCH
            cfg.ACTIVE.END_ITER *= cfg.DATALOADER.ITERS_PER_EPOCH
            cfg.ACTIVE.SAMPLE_ITER *= cfg.DATALOADER.ITERS_PER_EPOCH
            cfg.freeze()
        else:
            cfg.ACTIVE.START_ITER *= cfg.DATALOADER.ITERS_PER_EPOCH
            cfg.ACTIVE.END_ITER *= cfg.DATALOADER.ITERS_PER_EPOCH
            cfg.ACTIVE.SAMPLE_ITER *= cfg.DATALOADER.ITERS_PER_EPOCH

        self.sampler = Sampler(cfg, data_size=len(self._common_dataset.img_items))

    def could_active_sampling(self):
        return self.trainer.iter % self._cfg.ACTIVE.SAMPLE_ITER == 0 \
                and self.trainer.iter >= self._cfg.ACTIVE.START_ITER \
                and self.trainer.iter < self._cfg.ACTIVE.END_ITER
    def could_pseudo_labeling(self):
        return self.trainer.iter % self._cfg.PSEUDO.CLUSTER_ITER == 0 or self.trainer.iter == self.trainer.start_iter
    def before_step(self):
        if self.could_pseudo_labeling() or (self.trainer.iter % self._cfg.ACTIVE.SAMPLE_ITER == 0 \
                and self.trainer.iter >= self._cfg.ACTIVE.START_ITER):
            self._step_timer.reset()

            # get memory features
            self.get_memory_features()

            # generate pseudo labels and centers
            all_labels, all_centers, active_labels = self.update_labels()

            if self.could_pseudo_labeling():
                self._logger.info(f"Update unsupervised data loader.")
                self.update_train_loader(all_labels)
                # update memory labels
                self.update_memory_labels(all_labels)

            if self.trainer.iter % self._cfg.ACTIVE.SAMPLE_ITER == 0:
                if self._cfg.ACTIVE.BUILD_DATALOADER:
                    self._logger.info(f"Update active data loader.")
                    self.update_active_loader(active_labels)

            sec = self._step_timer.seconds()
            self._logger.info(f"Finished updating label in {str(datetime.timedelta(seconds=int(sec)))}")

    def update_labels(self):
        self._logger.info(f"Start updating labels on iteration {self.trainer.iter}")
        
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

        # calculate class-wise weight matrix
        class_weight_matrix = dist2classweight(dist_mat, num_classes, labels)
        cluster_num, _, _ = self.label_summary(labels, gt_labels, indep_thres=indep_thres)

        # active sampling
        if self.could_active_sampling() and self.sampler.could_sample():
            self.sampler.sample(dist_mat, labels, num_classes, gt_labels, cluster_num=cluster_num)

        # node label propagation
        if self._cfg.ACTIVE.NODE_PROP and len(self.sampler.query_set) > 0:
            active_labels = self.label_propagate(features.cpu().numpy(), gt_labels)
        else:
            labeled_index = list(self.sampler.query_set)
            gt_query = gt_labels[labeled_index].tolist()
            classes = list(set(gt_query))
            active_labels = [classes.index(gt_query[labeled_index.index(i)]) if i in labeled_index else -1 for i in range(len(labels))]

        # edge label propagation and rectification
        if self._cfg.ACTIVE.RECTIFY and len(self.sampler.query_set) > 0:
            labels, centers, num_classes, indep_thres, dist_mat, class_weight_matrix = self.rectify(features, labels, num_classes, indep_thres, dist_mat, gt_labels)
        
        if self._cfg.PSEUDO.MEMORY.WEIGHTED:
            # set weight matrix for calculating weighted contrastive loss.
            self.trainer.weight_matrix = class_weight_matrix

        all_labels.append(labels.tolist())
        all_centers.append(centers)

        self.indep_thres = indep_thres
        self.num_classes = num_classes

        return all_labels, all_centers, active_labels

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

    def rectify(self, features, labels, num_classes, indep_thres, dist_mat, gt_labels):
        if self._cfg.ACTIVE.EDGE_PROP:
            new_dist_mat = self.edge_label_propagation(dist_mat, gt_labels, max_iter=0, alpha=0.5)
        else:
            new_dist_mat = set_labeled_instances(self.sampler, dist_mat, gt_labels)

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
        self.trainer._data_loader_iter_active = iter(self.trainer.data_loader_active)
        # update cfg
        if self._cfg.is_frozen():
            self._cfg.defrost()
            self._cfg.MODEL.HEADS.NUM_CLASSES = self.trainer.data_loader_active.dataset.num_classes
            self._cfg.freeze()
        else:
            self._cfg.MODEL.HEADS.NUM_CLASSES = self.trainer.data_loader_active.dataset.num_classes

    def edge_label_propagation(self, dist_mat, gt_labels, max_iter=1, alpha=0.5):
        """
        edge label propagation. 
        Ensemble Diffusion for Retrieval, eq(3)
        A(t+1) = αS A(t) S.t() + (1 − α)I, S = D^(−1/2) W D^(−1/2), in our code W = A
        """
        # mask for residual regression
        mask = torch.zeros_like(dist_mat)
        for i in self.sampler.query_set:
            for j in self.sampler.query_set:
                mask[i, j] = 1
                mask[j, i] = 1
        dist_mat = dist_mat.cuda()
        adj = 1 - dist_mat
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        # numerical error
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        S = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
        # iterative propagation
        I = torch.eye(adj.size(0)).cuda()
        for _ in range(max_iter):
            adj = alpha * S @ adj @ S.T + (1 - alpha) * I
            adj = adj.clamp(0, 1)
        dist_mat = (1 - adj).cpu()
        return set_labeled_instances(self.sampler, dist_mat, gt_labels)
        
    def node_label_propagation(self, dist_mat):
        pass
    def label_propagate(self, X, true_labels, k = 50, max_iter = 20) -> list: 
        self._logger.info('Perform node label propagation')
        alpha = 0.99
        labels = np.asarray(true_labels)
        query_set = self.sampler.query_set
        labeled_idx = np.asarray(list(query_set))
        unlabeled_idx = np.asarray(list(set(range(len(labels))) - query_set))
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
            f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
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

        active_labels = p_labels.tolist()

        p_labels = np.array([classes[i] for i in p_labels])
        # Compute the accuracy of pseudo labels for statistical purposes
        correct_idx = (p_labels == labels)
        acc = correct_idx.mean()
        self._logger.info(f"label propagation acc: %.2f" % (acc * 100))
        return active_labels

class Sampler:
    def __init__(self, cfg, data_size):
        self._logger = logging.getLogger('fastreid.' + __name__)
        self.query_set = set()
        # sampling limitation
        self.M = int(cfg.ACTIVE.SAMPLE_M * data_size)
        self.query_sample_num = int(self.M * cfg.ACTIVE.SAMPLE_ITER // (cfg.ACTIVE.END_ITER - cfg.ACTIVE.START_ITER))
        # sampling method
        self.query_func = cfg.ACTIVE.SAMPLER.QUERY_FUNC
        # extra variables
        self.data_size = data_size

    def sample(self, dist_mat, pred_labels, pred_num_classes, targets, cluster_num=None):
        # get similarity matrix based on clusters
        clu_sim_mat = dist2classweight(dist_mat, pred_num_classes, pred_labels)
        if cluster_num:
            clu_sim_mat = clu_sim_mat[:, :cluster_num]
        query_index = self.query_sample(clu_sim_mat, dist_mat, pred_labels, cluster_num)
        self.query_set.update(query_index)
        self.query_summary(query_index, pred_labels, targets)

    def query_sample(self, clu_sim_mat, dist_mat, pred_labels, cluster_num, temp=0.05):
        sim_mat = clu_sim_mat / temp
        assert self.query_sample_num != 0
        query_index_list = list()

        if self.query_func == 'random':
            query_index = torch.randperm(self.data_size)
        elif self.query_func == 'entropy':
            sim_prob = sim_mat.softmax(dim=1)
            sim_entropy = (-sim_prob * (sim_prob + 1e-6).log()).sum(dim=1)
            query_index = sim_entropy.argsort(descending=True)
        elif self.query_func == 'confidence':
            sim_max_prob = sim_mat.max(dim=1)[0]
            query_index = sim_max_prob.argsort()
        elif self.query_func == 'diff':
            sim_sorted = sim_mat.sort(descending=True)[0]
            sim_diff = sim_sorted[:, 0] - sim_sorted[:, 1]
            query_index = sim_diff.argsort()
        elif self.query_fun == 'cluster':
            query_index = self.cluster_sample(dist_mat, pred_labels, cluster_num)
        else:
            raise NotImplemented(f"{self.query_func} is not supported in query selection.")
        
        for idx in query_index:
            if len(query_index_list) >= self.query_sample_num:
                break
            if idx not in self.query_set:
                query_index_list.append(idx.item())               

        return query_index_list
        
    def cluster_sample(self, dist_mat, pred_labels, cluster_num):
        pass
        # center_labels = list(collections.Counter(pred_labels.tolist()).items())
        # center_labels.sort(key=lambda d:d[1], reverse=True)
        
        # sel_clusters = [d[0] for d in center_labels[:cluster_num]]
        
        # dist = compute_distance_matrix(centers[sel_clusters], features)
        # indexes = dist.sort(dim=1)[1]

        # labeled_idx = []
        # for i, cluster in enumerate(sel_clusters):
        #     labeled_idx.append(indexes[i][:self.img_per_cluster])
        # labeled_idx = torch.cat(labeled_idx).tolist()
        # return labeled_idx

    def query_summary(self, query_index, pred_labels, targets):
        selected_pairs = [(i, j) for i in query_index for j in query_index if i!=j]
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
        self._logger.info(f'size of query_set is {len(self.query_set)}')
        # storage summary into tensorboard
        storage = get_event_storage()
        storage.put_scalar("tn", tn)
        storage.put_scalar("fp", fp)
        storage.put_scalar("fn", fn)
        storage.put_scalar("tp", tp)

    def could_sample(self):
        return len(self.query_set) < self.M


def dist2classweight(dist_mat, num_classes, labels):
    """
    calculate class-wise weight matrix from dist matrix.
    """
    weight_matrix = []
    for i in range(num_classes):
        index = torch.where(labels == i)[0].tolist()
        dist_ = dist_mat[:, index].mean(1)
        weight_matrix.append(dist_)
    weight_matrix = 1 - torch.stack(weight_matrix).t()
    return weight_matrix

def set_labeled_instances(sampler, dist_mat, gt_labels):
    new_dist_mat = dist_mat.clone()
    for i in sampler.query_set:
        for j in sampler.query_set:
            if gt_labels[i] == gt_labels[j]:
                new_dist_mat[i, j] = 0
                new_dist_mat[j, i] = 0
            else:
                new_dist_mat[i, j] = 1
                new_dist_mat[j, i] = 1
    return new_dist_mat