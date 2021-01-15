'''
Author: WuYiming
Date: 2020-10-12 21:22:11
LastEditTime: 2020-10-14 16:06:27
LastEditors: Please set LastEditors
Description: Hooks for SpCL
FilePath: /fast-reid/projects/SpCL_new/hooks.py
'''

import datetime
import itertools
import collections

import torch
import torch.nn.functional as F

from fastreid.utils import comm
from fastreid.engine.hooks import *
from fastreid.utils.metrics import euclidean_dist
from fastreid.utils.torch_utils import extract_features
from fastreid.utils.clustering import label_generator_dbscan, label_generator_kmeans

from fastreid.engine.train_loop import HookBase
from fastreid.data import build_reid_train_loader
from fastreid.data.samplers import InferenceSampler
from fastreid.data.build import fast_batch_collator
from fastreid.evaluation.evaluator import inference_context
from samplers import build_active_samplers
import random
import numpy as np
import faiss
from faiss import normalize_L2
import time
import scipy
import scipy.stats


class ActiveClusterHook(LabelGeneratorHook):

    def __init__(self, cfg, model, data_len):
        super().__init__(cfg, model)
        self.img_per_cluster = 2
        self.cluster_num = int(cfg.ACTIVE.SAMPLE_M * data_len) // self.img_per_cluster
        self.labeled_idx = None
        self.gt_set = None

    def before_step(self):
        if self.trainer.iter == self.trainer.start_iter:
            # generate pseudo labels and centers
            all_labels, all_centers = self.update_labels()
            self.label_propagate(self.features)
            comm.synchronize()
            
            sup_commdataset = self.trainer.data_loader.dataset
            sup_datasets = sup_commdataset.datasets

            re_gt_set = self.p_labels
            dataset = sup_datasets[0].data
            cam_labels = [dataset[i][2] for i in range(len(dataset))]
            
            # self.gt_dict = dict()
            # i = 0
            # for p in self.gt_set:
            #     if self.gt_dict.get(p) is None:
            #         self.gt_dict[p] = i
            #         i += 1
            # re_gt_set = [self.gt_dict[p] for p in self.gt_set]
            # dataset = sup_datasets[0].data
            # sup_datasets[0].data = [dataset[i] for i in self.labeled_idx]
            # cam_labels = [dataset[i][2] for i in self.labeled_idx]
            self.trainer.data_loader = build_reid_train_loader(self._cfg,
                                                           datasets=sup_datasets,
                                                           pseudo_labels=[re_gt_set],
                                                           cam_labels=[cam_labels],
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

    def update_labels(self):
        self._logger.info(f"Start updating pseudo labels on iteration {self.trainer.iter}")
        
        all_features = []
        features, true_labels = extract_features(self.model,
                                                self._data_loader_cluster,
                                                self._cfg.PSEUDO.NORM_FEAT)
        all_features.append(features)
        all_features = torch.stack(all_features, dim=0).mean(0)

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

            if comm.is_main_process():
                # clustering only on first GPU
                start_id, end_id = datasets_size_range[idx], datasets_size_range[idx + 1]
                labels, centers, num_classes, indep_thres, dist_mat = self.label_generator(
                    self._cfg,
                    all_features[start_id: end_id],
                    num_classes=num_classes,
                    indep_thres=indep_thres
                )
                if self._cfg.PSEUDO.NORM_CENTER:
                    centers = F.normalize(centers, p=2, dim=1)

                from collections import Counter
                center_labels = list(Counter(labels.tolist()).items())
                center_labels.sort(key=lambda d:d[1], reverse=True)

                sel_clusters = [d[0] for d in center_labels[:self.cluster_num]]
                
                dist = euclidean_dist(centers[sel_clusters], features)
                indexes = dist.sort(dim=1)[1]

                labeled_idx = []
                for i, cluster in enumerate(sel_clusters):
                    labeled_idx.append(indexes[i][:self.img_per_cluster])
                labeled_idx = torch.cat(labeled_idx).tolist()
                
                self.labeled_idx = labeled_idx
                self.gt_set = true_labels[labeled_idx].tolist()
                self.true_labels = true_labels.tolist()
                self.features = features.cpu().numpy()
                
            comm.synchronize()

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

    def label_propagate(self, X, k = 50, max_iter = 20):
        
        self._logger.info('Updating pseudo-labels...')
        alpha = 0.99
        labels = np.asarray(self.true_labels)
        labeled_idx = np.asarray(self.labeled_idx)
        self.unlabeled_idx = list(set(range(len(labels))) - set(self.labeled_idx))
        unlabeled_idx = np.asarray(self.unlabeled_idx)
        classes = np.unique(self.gt_set)

        # kNN search for the graph
        d = X.shape[1]
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = int(torch.cuda.device_count()) - 1
        index = faiss.GpuIndexFlatIP(res,d,flat_config)   # build the index

        normalize_L2(X)
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
        
        self.p_labels = p_labels.tolist()

        p_labels = np.array([classes[i] for i in p_labels])
        # Compute the accuracy of pseudolabels for statistical purposes
        correct_idx = (p_labels == labels)
        acc = correct_idx.mean()
        self._logger.info(f"label propagation acc: %.2f" % (acc * 100))


class ActiveHook(HookBase):

    def __init__(self, cfg, data_len):
        super().__init__()
        self.base_iter = cfg.ACTIVE.TRAIN_ITER * cfg.DATALOADER.ITERS_PER_EPOCH
        index_set = list(range(data_len))
        random.shuffle(index_set)
        index_set = index_set[:int(data_len * cfg.ACTIVE.SAMPLE_M)]
        self.index_set = index_set
        self.samplers = build_active_samplers(cfg, index_set)
        labeled_num = int(len(index_set) * cfg.ACTIVE.INITIAL_RATE) + 1
        index_dataloader = torch.utils.data.DataLoader(index_set, batch_size=labeled_num, shuffle=True)
        self._index_iter = iter(index_dataloader)
        self.warmup_iters = cfg.ACTIVE.WARMUP_ITER * cfg.DATALOADER.ITERS_PER_EPOCH
        self.active_max_iters = self.base_iter / cfg.ACTIVE.INITIAL_RATE + self.warmup_iters

    def before_step(self):
        if self.warmup_iters <= self.trainer.iter < self.active_max_iters and self.trainer.iter % self.base_iter == 0:
            indexes = self._index_iter.next()
            all_features, targets = self.get_feature()
            features = all_features[self.index_set]
            sel_feats = all_features[indexes]
            dist_mat = self.euclidean_dist(sel_feats, features)
            # only choose first 30 similar instances
            sim_mat = torch.argsort(dist_mat, dim=1)[:, 1:31]
            self.samplers.sample(indexes, sim_mat, targets)
            self.trainer.data_loader_active = self.trainer.build_active_sample_dataloader(self.samplers.triplet_set, is_train=True)
            self.trainer._data_loader_active_iter = iter(self.trainer.data_loader_active)
            self.trainer.active_warmup = True
            
    def get_feature(self):
        num_workers = self.trainer.cfg.DATALOADER.NUM_WORKERS
        batch_size = self.trainer.cfg.TEST.IMS_PER_BATCH
        data_sampler = InferenceSampler(len(self.trainer.data_loader.dataset))
        batch_sampler = torch.utils.data.BatchSampler(data_sampler, batch_size, False)
        dataloader = torch.utils.data.DataLoader(
            self.trainer.data_loader.dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=fast_batch_collator,
        )

        features = []
        targets = []
        model = self.trainer.model
        with inference_context(model), torch.no_grad():
            for idx, inputs in enumerate(dataloader):
                outputs = model(inputs)
                features.append(outputs)
                targets.append(inputs['targets'])
            features = torch.cat(features)
            targets = torch.cat(targets)
        return features, targets
    
    def euclidean_dist(self, x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(x, y.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist