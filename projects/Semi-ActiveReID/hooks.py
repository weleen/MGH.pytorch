import datetime
import itertools
import logging
import os
import time
import numpy as np
import collections
import faiss
import scipy

import torch
import torch.nn.functional as F

from fastreid.utils import comm, graph
from fastreid.engine.hooks import LabelGeneratorHook
from fastreid.data import build_reid_train_loader
from fastreid.utils.torch_utils import extract_features
from fastreid.utils.faiss_utils import search_raw_array_pytorch

import sampling_method

class SALLabelGeneratorHook(LabelGeneratorHook):
    """ Hook for the combination of unsupervised learning and active learning.
    """
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        try:
            self.sampler = getattr(sampling_method, cfg.ACTIVE.QUERY_FUNC)(cfg, dataset=self._common_dataset.img_items)
        except Exception as e:
            self._logger.info('Error when execute {}'.format(cfg.ACTIVE.QUERY_FUNC))
            raise

    def could_active_sampling(self):
        return self.trainer.epoch % self._cfg.ACTIVE.SAMPLE_EPOCH == 0 \
                and self.trainer.epoch >= self._cfg.ACTIVE.START_EPOCH \
                and self.trainer.epoch < self._cfg.ACTIVE.END_EPOCH \
                and (self._cfg.ACTIVE.RECTIFY or self._cfg.ACTIVE.BUILD_DATALOADER)

    def could_pseudo_labeling(self):
        return self.trainer.epoch % self._cfg.PSEUDO.CLUSTER_EPOCH == 0 \
            or self.trainer.epoch == self.trainer.start_epoch

    def before_epoch(self):
        if self.could_pseudo_labeling() or self.could_active_sampling():
            self._step_timer.reset()

            # get memory features
            self.get_memory_features()

            # generate pseudo labels and centers
            all_labels, all_centers, active_labels = self.update_labels()

            self._logger.info(f"Update unsupervised data loader.")
            self.update_train_loader(all_labels)

            # reset optimizer
            if self._cfg.PSEUDO.RESET_OPT:
                self._logger.info(f"Reset optimizer")
                self.trainer.optimizer.state = collections.defaultdict(dict)

            if hasattr(self.trainer, 'memory'):
                # update memory labels, memory based methods such as SpCL
                self.update_memory_labels(all_labels)
                assert len(all_centers) == 1, 'only support single unsupervised dataset'
                self.trainer.memory._update_center(all_centers[0])
            else:
                # update classifier centers, methods such as SBL
                self.update_classifier_centers(all_centers)

            if self.could_active_sampling() and self._cfg.ACTIVE.BUILD_DATALOADER:
                self._logger.info(f"Update active data loader.")
                self.update_active_loader(active_labels)

            comm.synchronize()

            sec = self._step_timer.seconds()
            self._logger.info(f"Finished updating pseudo label in {str(datetime.timedelta(seconds=int(sec)))}")

    def update_labels(self):
        self._logger.info(f"Start updating pseudo labels on epoch {self.trainer.epoch}/iteration {self.trainer.iter}")
        
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
        all_active_labels = []
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
                save_path = '{}/clustering/clustering_epoch{}.pt'.format(self._cfg.OUTPUT_DIR, self.trainer.epoch)
                start_id, end_id = datasets_size_range[idx], datasets_size_range[idx + 1]
                labels, centers, num_classes, indep_thres, dist_mat = self.label_generator(
                    self._cfg,
                    all_features[start_id: end_id],
                    num_classes=num_classes,
                    indep_thres=indep_thres,
                    epoch=self.trainer.epoch
                )
                if not os.path.exists(save_path):
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    res = {'labels': labels, 'num_classes': num_classes}#, 'centers': centers, 'indep_thres': indep_thres, 'dist_mat': dist_mat}
                    torch.save(res, save_path)

                if self._cfg.PSEUDO.NORM_CENTER:
                    centers = F.normalize(centers, p=2, dim=1)

                features = all_features[start_id: end_id]
                gt_labels = true_labels[start_id: end_id]
                # calculate weight_matrix if use weight_matrix in loss calculation
                class_weight_matrix = sampling_method.dist2classweight(dist_mat, num_classes, labels)
                cluster_num, _, _ = self.label_summary(labels, gt_labels, indep_thres=indep_thres)

                # active sampling 
                if self.could_active_sampling() and self.sampler.could_sample():
                    self.sampler.query(dist_mat, labels, num_classes, gt_labels, cluster_num=cluster_num, features=features, centers=centers)

                # # node label propagation
                # if self._cfg.ACTIVE.NODE_PROP and self.sampler.index_label.sum() > 0:
                #     active_labels = self.node_label_propagation(features.cpu().numpy(), gt_labels)
                #     classes = list(set(active_labels))
                #     active_labels = [classes.index(i) for i in active_labels]
                # else:
                labeled_index = list(self.sampler.query_set)
                gt_query = gt_labels[labeled_index].tolist()
                classes = list(set(gt_query))
                # reset the labels
                active_labels = [classes.index(gt_query[labeled_index.index(i)]) if i in labeled_index else -1 for i in range(len(labels))]

                # edge label propagation and rectification
                if self._cfg.ACTIVE.RECTIFY and self.sampler.index_label.sum() > 0:
                    labels, centers, num_classes, indep_thres, dist_mat, class_weight_matrix = self.rectify(features, labels, num_classes, indep_thres, dist_mat, gt_labels)
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
                dist_mat = comm.broadcast_tensor(dist_mat, 0)
                active_labels = comm.broadcast_tensor(active_labels, 0)

            if self._cfg.PSEUDO.MEMORY.WEIGHTED:
                self.trainer.weight_matrix = class_weight_matrix
            all_labels.append(labels.tolist())
            all_centers.append(centers)
            all_active_labels.append(active_labels)

            try:
                self.indep_thres[idx] = indep_thres
            except:
                self.indep_thres.append(indep_thres)
            try:
                self.num_classes[idx] = num_classes
            except:
                self.num_classes.append(num_classes)

        return all_labels, all_centers, all_active_labels

    def rectify(self, features, labels, num_classes, indep_thres, dist_mat, gt_labels):
        if self._cfg.ACTIVE.EDGE_PROP:
            new_dist_mat = self.edge_label_propagation_(features, dist_mat, gt_labels, step=self._cfg.ACTIVE.EDGE_PROP_STEP)
            torch.cuda.empty_cache()
        else:
            new_dist_mat = sampling_method.set_labeled_instances(self.sampler, dist_mat, gt_labels)

        new_labels, new_centers, new_num_classes, new_indep_thres, new_dist_mat = self.label_generator(
                    self._cfg,
                    features,
                    num_classes=num_classes,
                    indep_thres=indep_thres,
                    dist=new_dist_mat.numpy()
                )
        if self._cfg.PSEUDO.NORM_CENTER:
            new_centers = F.normalize(new_centers, p=2, dim=1)

        new_weight_matrix = sampling_method.dist2classweight(new_dist_mat, new_num_classes, new_labels)
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

    def edge_label_propagation(self, features: torch.Tensor, dist_mat: torch.Tensor, gt_labels: torch.Tensor, step=50, alpha=0.99, debug=False):
        # TODO: try cdp graph propagation
        """
        edge label propagation. 
        Ensemble Diffusion for Retrieval, eq(3)
        A(t+1) = αS A(t) S.t() + (1 − α)I, S = D^(−1/2) W D^(−1/2), in our code W = A
        """
        start_time = time.time()
        # mask for residual regression
        dist_mat = dist_mat.cuda('cuda:1')
        corrected_dist_mat, labeled_dist_mat = sampling_method.set_labeled_instances(self.sampler, dist_mat, gt_labels)
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

    def edge_label_propagation_(self, features: torch.Tensor, dist_mat: torch.Tensor, gt_labels: torch.Tensor, k=15, th=0.66, step=0.05, max_iter=100, max_sz=600, debug=False, **kwargs):
        if dist_mat is not None:
            knn_dist, knn_idx = torch.sort(dist_mat, dim=1)
            knn_dist = knn_dist[:, :k].cpu().numpy()
            knn_idx = knn_idx[:, :k].cpu().numpy()
        else:
            res = faiss.StandardGpuResources()
            res.setDefaultNullStreamAllDevices()
            knn_dist, knn_idx = search_raw_array_pytorch(res, features, features, k)  # normalized features, jaccard distances
            knn_dist = knn_dist.cpu().numpy()
            knn_idx = knn_idx.cpu().numpy()

        # generate pairs and scores
        simi = 1. - knn_dist
        anchor = np.tile(np.arange(len(knn_idx)).reshape(len(knn_idx), 1), (1, knn_idx.shape[1]))
        selidx = np.where((simi > th) & (knn_idx != -1) & (knn_idx != anchor))

        pairs = np.hstack((anchor[selidx].reshape(-1, 1), knn_idx[selidx].reshape(-1, 1)))
        scores = simi[selidx]
        pairs = np.sort(pairs, axis=1)
        pairs, unique_idx = np.unique(pairs, return_index=True, axis=0)
        scores = scores[unique_idx]
        import time
        a = time.time()
        # generate positive pairs and negative pairs from sampler indexes
        if self.sampler.index_label.sum() > 0:
            pos_pairs = []
            neg_pairs = []
            for i, j in itertools.permutations(self.sampler.query_set, 2):
                if gt_labels[i] == gt_labels[j]:
                    # positive pairs
                    pos_pairs.append([i, j])
                    if [i, j] in pairs:
                        index = np.where((pairs == (i, j)).all(axis=1))[0]
                        scores[index] = 1.0
                    else:
                        pairs = np.concatenate((pairs, np.array([[i, j]])))
                        scores = np.concatenate((scores, np.array([1.0])))
                else:
                    # negative pairs
                    neg_pairs.append([i, j])
                    if [i, j] in pairs:
                        index = np.where((pairs == (i, j)).all(axis=1))[0]
                        scores[index] = 0.0
            kept_index = np.where(scores > th)[0]
            pairs = pairs[kept_index]
            scores = scores[kept_index]
        print('{}'.format(time.time() - a))
        components = graph.graph_propagation(pairs, scores, max_sz, step, max_iter)

        # collect results
        cdp_res = []
        for c in components:
            cdp_res.append(sorted([n.name for n in c]))
        pred = -1 * np.ones(len(features), dtype=np.int)
        for i, c in enumerate(cdp_res):
            pred[np.array(c)] = i

        valid = np.where(pred != -1)
        _, unique_idx = np.unique(pred[valid], return_index=True)
        pred_unique = pred[valid][np.sort(unique_idx)]
        pred_mapping = dict(zip(list(pred_unique), range(pred_unique.shape[0])))
        pred_mapping[-1] = -1
        pred = np.array([pred_mapping[p] for p in pred])

        outlier_index = np.where(pred == -1)[0]
        pred_max = pred.max() + 1
        pred_classes = np.arange(pred_max, pred_max + len(outlier_index))
        pred[outlier_index] = pred_classes
        
        new_dist_mat = 1 - (pred.reshape(-1, 1) == pred.reshape(1, -1))
        new_dist_mat = torch.from_numpy(new_dist_mat).to(dist_mat.device)
        if dist_mat is not None:
            new_dist_mat = (new_dist_mat + dist_mat) / 2
        return new_dist_mat