# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import os
import copy
import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

from .evaluator import DatasetEvaluator
from .query_expansion import aqe
from .rank import evaluate_rank
from .roc import evaluate_roc
from fastreid.utils import comm
from fastreid.utils.metrics import compute_distance_matrix

logger = logging.getLogger(__name__)

def joint_similarity(q_cam, q_frame, g_cam, g_frame, distribution):
    interval = 100
    score_st = np.zeros((len(q_cam), len(g_cam)))
    for i in range(len(q_cam)):
        for j in range(len(g_cam)):
            if q_frame[i] > g_frame[j]:
                diff = q_frame[i] - g_frame[j]
                hist_ = int(diff / interval)
                pr = distribution[q_cam[i]][g_cam[j]][hist_]
            else:
                diff = g_frame[j] - q_frame[i]
                hist_ = int(diff / interval)
                pr = distribution[g_cam[j]][q_cam[i]][hist_]
            score_st[i][j] = pr

    return score_st

class ReidEvaluator(DatasetEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        self.cfg = cfg
        self._num_query = num_query
        self._output_dir = output_dir

        self.features = []
        self.pids = []
        self.camids = []
        self.frameids = []

    def reset(self):
        self.features = []
        self.pids = []
        self.camids = []
        self.frameids = []

    def process(self, inputs, outputs):
        self.pids.extend(inputs["targets"])
        self.camids.extend(inputs["camids"])
        self.features.append(outputs.data.cpu())
        if self.cfg.DATASETS.NAMES[0] == "Market1501":
            self.frameids.extend([int(path.split('/')[-1].split('_')[-2]) for path in inputs["img_paths"]])
        elif self.cfg.DATASETS.NAMES[0] == "DukeMTMC":
            self.frameids.extend([int(path.split('/')[-1][9:16]) for path in inputs["img_paths"]])

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            features = comm.gather(self.features)
            features = sum(features, [])

            pids = comm.gather(self.pids)
            pids = sum(pids, [])

            camids = comm.gather(self.camids)
            camids = sum(camids, [])

            # fmt: off
            if not comm.is_main_process(): return {}
            # fmt: on
        else:
            features = self.features
            pids = self.pids
            camids = self.camids
            frameids = self.frameids

        features = torch.cat(features, dim=0)
        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = np.asarray(pids[:self._num_query])
        query_camids = np.asarray(camids[:self._num_query])
        query_frameids = np.asarray(frameids[:self._num_query])

        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_pids = np.asarray(pids[self._num_query:])
        gallery_camids = np.asarray(camids[self._num_query:])
        gallery_frameids = np.asarray(frameids[self._num_query:])

        self._results = OrderedDict()

        if self.cfg.TEST.AQE.ENABLED:
            logger.info("Test with AQE setting")
            qe_time = self.cfg.TEST.AQE.QE_TIME
            qe_k = self.cfg.TEST.AQE.QE_K
            alpha = self.cfg.TEST.AQE.ALPHA
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

        dist = compute_distance_matrix(query_features, gallery_features, self.cfg.TEST.METRIC)

        if self.cfg.CAP.ST_TEST:
            distribution = np.load(os.path.join(self.cfg.OUTPUT_DIR, 'distribution.npy'))
            score_st = joint_similarity(query_camids, query_frameids, gallery_camids, gallery_frameids, distribution)

            gamma = 5
            dist = 1 - 1 / (1 + np.exp(-gamma * (1 - dist))) / (1 + 2 * np.exp(-gamma * score_st))

        if self.cfg.TEST.RERANK.ENABLED:
            logger.info("Test with rerank setting")
            k1 = self.cfg.TEST.RERANK.K1
            k2 = self.cfg.TEST.RERANK.K2
            lambda_value = self.cfg.TEST.RERANK.LAMBDA

            if self.cfg.TEST.METRIC == "cosine":
                query_features = F.normalize(query_features, dim=1)
                gallery_features = F.normalize(gallery_features, dim=1)

            if self.cfg.TEST.RERANK.TYPE in ['gnn', 'jaccard']:
                rerank_dist = compute_distance_matrix(query_features, gallery_features, metric=self.cfg.TEST.RERANK.TYPE, k1=k1, k2=k2)
            else:
                raise ValueError(f'Unsupported value {self.cfg.TEST.RERANK.TYPE}, select from [gnn | jaccard]')
            dist = rerank_dist * (1 - lambda_value) + dist * lambda_value

        logger.info("Calculate CMC and mAP")
        cmc, all_AP, all_INP = evaluate_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids)

        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1]
        self._results['mAP'] = mAP
        self._results['mINP'] = mINP

        if self.cfg.TEST.ROC_ENABLED:
            logger.info("Calculate ROC")
            scores, labels = evaluate_roc(dist, query_pids, gallery_pids, query_camids, gallery_camids)
            fprs, tprs, thres = metrics.roc_curve(labels, scores)

            for fpr in [1e-4, 1e-3, 1e-2]:
                ind = np.argmin(np.abs(fprs - fpr))
                self._results["TPR@FPR={:.0e}".format(fpr)] = tprs[ind]

        return copy.deepcopy(self._results)
