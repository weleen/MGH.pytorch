import logging
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
from fastreid.utils.events import get_event_storage


class Sampler:
    def __init__(self, cfg, dataset):
        self._logger = logging.getLogger('fastreid.' + __name__)
        self.dataset = dataset
        self.data_size = len(dataset)
        self.query_set = set()
        self.index_label = np.zeros(self.data_size, dtype=bool)
        # sampling limitation
        self.M = int(cfg.ACTIVE.SAMPLE_M * self.data_size)
        self.query_num = int(self.M * cfg.ACTIVE.SAMPLE_EPOCH // (cfg.ACTIVE.END_EPOCH - cfg.ACTIVE.START_EPOCH))
        assert self.query_num != 0
        self.query_func = cfg.ACTIVE.QUERY_FUNC

    def query(self, dist_mat, pred_labels, pred_num_classes, targets, cluster_num=None, features=None, centers=None, temp=0.05) -> None:
        query_index = self.query_sample(dist_mat, pred_labels, pred_num_classes, targets, cluster_num, features, centers, temp)
        self.index_label[query_index] = True
        self.query_set.update(query_index)
        self.query_summary(query_index, pred_labels, targets)

    def query_sample(self, dist_mat, pred_labels, pred_num_classes, targets, cluster_num=None, features=None, centers=None, temp=0.05) -> np.ndarray:
        raise NotImplementedError

    def query_summary(self, query_index, pred_labels, targets):
        selected_pairs = list(itertools.permutations(query_index, 2))
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
