import torch
import logging
import collections
import numpy as np

from sklearn.metrics import confusion_matrix
from fastreid.utils.events import get_event_storage
from fastreid.utils.metrics import compute_distance_matrix


class InstaceSampler:
    def __init__(self, cfg, dataset):
        self._logger = logging.getLogger('fastreid.' + __name__)
        self.dataset = dataset
        self.data_size = len(dataset)
        self.query_set = set()
        self.index_label = np.zeros(self.data_size, dtype=bool)
        # sampling limitation
        self.M = int(cfg.ACTIVE.INSTANCE.SAMPLE_M * self.data_size)
        self.query_num = int(self.M * cfg.ACTIVE.SAMPLE_EPOCH // (cfg.ACTIVE.END_EPOCH - cfg.ACTIVE.START_EPOCH))
        assert self.query_num != 0
        self.query_func = cfg.ACTIVE.INSTANCE.QUERY_FUNC

    def query_sample(self, dist_mat, pred_labels, pred_num_classes, targets, cluster_num=None, features=None, centers=None, temp=0.05) -> None:
        # get similarity matrix based on clusters
        sim_mat = dist2classweight(dist_mat, pred_num_classes, pred_labels) / temp
        if cluster_num:
            sim_mat = sim_mat[:, :cluster_num]

        if self.query_func == 'random':
            indexes = np.where(self.index_label==0)[0]
            query_index = indexes[np.random.permutation(len(indexes))][:self.query_num]
        elif self.query_func == 'confidence':  # query if highest confidence is low
            ulbl_indexes = np.arange(self.data_size)[~self.index_label]
            probs = sim_mat[ulbl_indexes]
            U = probs.max(dim=1)[0]
            query_index = ulbl_indexes[U.sort()[1][:self.query_num]]
        elif self.query_func == 'margin':  # query if margin between top-2 is small
            ulbl_indexes = np.arange(self.data_size)[~self.index_label]
            probs = sim_mat[ulbl_indexes]
            probs_sorted = probs.sort(descending=True)[0]
            U = probs_sorted[:, 0] - probs_sorted[:, 1]
            query_index = ulbl_indexes[U.sort()[1][:self.query_num]]
        elif self.query_func == 'entropy':  # query if entropy is high
            ulbl_indexes = np.arange(self.data_size)[~self.index_label]
            probs = sim_mat[ulbl_indexes]
            log_probs = torch.log(probs)
            U = (probs*log_probs).sum(1)
            query_index = ulbl_indexes[U.sort()[1][:self.query_num]]
        elif self.query_func == 'coreset': # k-center
            query_index = self.coreset_sample(self, dist_mat, pred_labels, pred_num_classes, targets, cluster_num, features, centers)
        elif self.query_func == 'cluster':  # selec instances based on clustering
            query_index = self.cluster_sample(pred_labels, features, centers, cluster_num)
        else:
            raise NotImplemented(f"{self.query_func} is not supported in query selection.")

        self.index_label[query_index] = True
        self.query_set.update(query_index)
        self.query_summary(query_index, pred_labels, targets)
        
    def coreset_sample(self, dist_mat):
        ulbl_indexes = np.arange(self.data_size)[~self.index_label]
        lbl_indexes = np.arange(self.data_size)[self.index_label]
        m = ulbl_indexes.shape[0]
        
        if len(lbl_indexes) == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = dist_mat[ulbl_indexes, :][:, lbl_indexes].numpy()
            # dist_ctr = compute_distance_matrix(ulbl_features, lbl_features, metric='cosine').numpy()
            min_dist = np.amin(dist_ctr, axis=1)

        query_index = []
        for _ in range(self.query_num):
            idx = min_dist.argmax()
            query_index.append(idx)
            new_lbl_indexes = np.append(lbl_indexes, idx)
            dist_new_ctr = dist_mat[ulbl_indexes, :][:, new_lbl_indexes].numpy()
            min_dist = np.amin(dist_new_ctr, axis=1)

        return ulbl_indexes[query_index]

    def cluster_sample(self, pred_labels, features, centers, cluster_num):
        # pass
        img_per_cluster = 1
        center_labels = list(collections.Counter(pred_labels.tolist()).items())
        center_labels.sort(key=lambda d:d[1], reverse=True)
        
        sel_clusters = [d[0] for d in center_labels[:cluster_num]]
        
        dist = compute_distance_matrix(centers[sel_clusters], features)
        indexes = dist.sort(dim=1)[1]

        labeled_idx = []
        for i, cluster in enumerate(sel_clusters):
            labeled_idx.append(indexes[i][:img_per_cluster])
        labeled_idx = torch.cat(labeled_idx)
        return labeled_idx

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
    weight_matrix = torch.zeros((len(labels), num_classes))
    nums = torch.zeros((1, num_classes))
    if labels.min() < 0:
        index_select = torch.where(labels >= 0)[0]
        inputs_select = dist_mat[index_select]
        labels_select = labels[index_select]
        weight_matrix.index_add_(1, labels_select, inputs_select)
        nums.index_add_(1, labels_select, torch.ones(1, len(index_select)))
    else:
        weight_matrix.index_add_(1, labels, dist_mat)
        nums.index_add_(1, labels, torch.ones(1, len(labels)))
    weight_matrix = 1 -  weight_matrix / nums
    return weight_matrix

def set_labeled_instances(sampler, dist_mat, gt_labels):
    new_dist_mat = dist_mat
    labeled_dist_mat = -torch.ones_like(dist_mat)
    for i in sampler.query_set:
        for j in sampler.query_set:
            if gt_labels[i] == gt_labels[j]:
                new_dist_mat[i, j] = 0
                new_dist_mat[j, i] = 0
                labeled_dist_mat[i, j] = 0
                labeled_dist_mat[j, i] = 0
            else:
                new_dist_mat[i, j] = 1
                new_dist_mat[j, i] = 1
                labeled_dist_mat[i, j] = 1
                labeled_dist_mat[j, i] = 1
    return new_dist_mat, labeled_dist_mat


def set_labeled_instances_np(sampler, dist_mat, gt_labels):
    new_dist_mat = dist_mat.copy()
    labeled_dist_mat = -np.ones_like(dist_mat)
    for i in sampler.query_set:
        for j in sampler.query_set:
            if gt_labels[i] == gt_labels[j]:
                new_dist_mat[i, j] = 0
                new_dist_mat[j, i] = 0
                labeled_dist_mat[i, j] = 0
                labeled_dist_mat[j, i] = 0
            else:
                new_dist_mat[i, j] = 1
                new_dist_mat[j, i] = 1
                labeled_dist_mat[i, j] = 1
                labeled_dist_mat[j, i] = 1
    return new_dist_mat, labeled_dist_mat