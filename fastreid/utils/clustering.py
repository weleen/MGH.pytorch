import logging
import faiss
import torch
import collections
import numpy as np
import torch
from sklearn.cluster import DBSCAN

from fastreid.utils.torch_utils import to_numpy, to_torch
from fastreid.utils.metrics import compute_distance_matrix

logger = logging.getLogger(__name__)

__all__ = ["label_generator_dbscan_single", "label_generator_dbscan", "label_generator_kmeans"]


@torch.no_grad()
def label_generator_kmeans(cfg, features, num_classes=500, cuda=True, **kwargs):
    assert cfg.PSEUDO.NAME == "kmeans"
    assert num_classes, "num_classes for kmeans is null"

    if not cfg.PSEUDO.USE_OUTLIERS:
        logger.warning("there exists no outlier point by kmeans clustering")

    # k-means cluster by faiss
    cluster = faiss.Kmeans(features.size(-1), num_classes, niter=300, verbose=True, gpu=cuda)
    cluster.train(to_numpy(features))
    centers = to_torch(cluster.centroids).float()
    _, labels = cluster.index.search(to_numpy(features), 1)
    labels = labels.reshape(-1)
    labels = to_torch(labels).long()
    # k-means does not have outlier points
    assert not (-1 in labels)

    return labels, centers, num_classes, None


@torch.no_grad()
def label_generator_dbscan_single(cfg, features, dist, eps, **kwargs):
    assert isinstance(dist, np.ndarray)

    min_samples = cfg.PSEUDO.DBSCAN.MIN_SAMPLES
    use_outliers = cfg.PSEUDO.USE_OUTLIERS

    cluster = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1, )
    labels = cluster.fit_predict(dist)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # cluster labels -> pseudo labels
    centers = collections.defaultdict(list)
    outliers = 0
    for i, label in enumerate(labels):
        if label == -1:
            if not use_outliers:
                continue
            labels[i] = num_clusters + outliers
            outliers += 1
        centers[labels[i]].append(features[i])

    centers = [torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())]
    centers = torch.stack(centers, dim=0)
    labels = to_torch(labels).long()
    num_clusters += outliers

    return labels, centers, num_clusters


@torch.no_grad()
def label_generator_dbscan(cfg, features, indep_thres=None, **kwargs):
    assert cfg.PSEUDO.NAME == "dbscan"

    dist = compute_distance_matrix(features,
                                   features,
                                   metric=cfg.PSEUDO.DBSCAN.DIST_METRIC,
                                   min_samples=cfg.PSEUDO.MIN_SAMPLES,
                                   k1=cfg.PSEUDO.K1,
                                   k2=cfg.PSEUDO.K2,
                                   search_type=cfg.PSEUDO.SEARCH_TYPE)
    features = features.cpu()
    # clustering
    eps = cfg.PSEUDO.DBSCAN.EPS

    if len(eps) == 1:
        # normal clustering
        labels, centers, num_classes = label_generator_dbscan_single(cfg, features, dist, eps[0])
        return labels, centers, num_classes, indep_thres
    else:
        assert len(eps) == 3, "three eps values are required for the clustering reliability criterion"

        logger.info("adopt the reliability criterion for filtering clusters")
        eps = sorted(eps)
        labels_tight, _, _ = label_generator_dbscan_single(cfg, features, dist, eps[0])
        labels_normal, _, num_classes = label_generator_dbscan_single(cfg, features, dist, eps[1])
        labels_loose, _, _ = label_generator_dbscan_single(cfg, features, dist, eps[2])

        # compute R_indep and R_comp
        N = labels_normal.size(0)
        label_sim = labels_normal.expand(N, N).eq(labels_normal.expand(N, N).t()).float()
        label_sim_tight = labels_tight.expand(N, N).eq(labels_tight.expand(N, N).t()).float()
        label_sim_loose = labels_loose.expand(N, N).eq(labels_loose.expand(N, N).t()).float()

        R_comp = 1 - torch.min(label_sim, label_sim_tight).sum(-1) / torch.max(label_sim, label_sim_tight).sum(-1)
        R_indep = 1 - torch.min(label_sim, label_sim_loose).sum(-1) / torch.max(label_sim, label_sim_loose).sum(-1)
        assert (R_comp.min() >= 0) and (R_comp.max() <= 1)
        assert (R_indep.min() >= 0) and (R_indep.max() <= 1)

        cluster_R_comp, cluster_R_indep = collections.defaultdict(list), collections.defaultdict(list)
        cluster_img_num = collections.defaultdict(int)
        for comp, indep, label in zip(R_comp, R_indep, labels_normal):
            cluster_R_comp[label.item()].append(comp.item())
            cluster_R_indep[label.item()].append(indep.item())
            cluster_img_num[label.item()] += 1

        cluster_R_comp = [min(cluster_R_comp[i]) for i in sorted(cluster_R_comp.keys())]
        cluster_R_indep = [min(cluster_R_indep[i]) for i in sorted(cluster_R_indep.keys())]
        cluster_R_indep_noins = [iou for iou, num in zip(cluster_R_indep, sorted(cluster_img_num.keys()))
                                 if cluster_img_num[num] > 1]
        if indep_thres is None:
            indep_thres = np.sort(cluster_R_indep_noins)[
                min(len(cluster_R_indep_noins) - 1, np.round(len(cluster_R_indep_noins) * 0.9).astype("int"))
            ]

        labels_num = collections.defaultdict(int)
        for label in labels_normal:
            labels_num[label.item()] += 1

        centers = collections.defaultdict(list)
        outliers = 0
        for i, label in enumerate(labels_normal):
            label = label.item()
            indep_score = cluster_R_indep[label]
            comp_score = R_comp[i]
            if label == -1:
                assert not cfg.PSEUDO.USE_OUTLIERS, "exists a bug"
                continue
            if (indep_score > indep_thres) or (comp_score.item() > cluster_R_comp[label]):
                if labels_num[label] > 1:
                    labels_normal[i] = num_classes + outliers
                    outliers += 1
                    labels_num[label] -= 1
                    labels_num[labels_normal[i].item()] += 1

            centers[labels_normal[i].item()].append(features[i])

        num_classes += outliers
        assert len(centers.keys()) == num_classes

        centers = [torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())]
        centers = torch.stack(centers, dim=0)

        return labels_normal, centers, num_classes, indep_thres
