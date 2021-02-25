import time
import logging
import faiss
import torch
import collections
import numpy as np
import torch
from sklearn.cluster import DBSCAN

from fastreid.utils.torch_utils import to_numpy, to_torch
from fastreid.utils.metrics import compute_distance_matrix
from fastreid.utils.graph import graph_propagation
from fastreid.utils.get_st_matrix import get_st_matrix
from fastreid.evaluation.rerank import re_ranking_dist
from hyperg import gen_knn_hg, gen_clustering_hg, concat_multi_hg, spectral_hg_partitioning
from scipy.sparse.linalg.eigen.arpack import eigsh


preH = None

logger = logging.getLogger(__name__)

__all__ = ["label_generator_dbscan_single", "label_generator_dbscan", "label_generator_kmeans", "label_generator_cdp", "label_generator_hypergraph"]


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

    return labels, centers, num_classes, None, None


@torch.no_grad()
def label_generator_dbscan_single(cfg, features, dist, eps, **kwargs):
    assert isinstance(dist, np.ndarray)

    min_samples = cfg.PSEUDO.DBSCAN.MIN_SAMPLES
    use_outliers = cfg.PSEUDO.USE_OUTLIERS

    cluster = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1)
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
    assert "dbscan" in cfg.PSEUDO.NAME

    if 'dist' in kwargs:
        dist = kwargs['dist']
    else:
        dist = compute_distance_matrix(features,
                                       None,
                                       metric=cfg.PSEUDO.DBSCAN.DIST_METRIC,
                                       k1=cfg.PSEUDO.DBSCAN.K1,
                                       k2=cfg.PSEUDO.DBSCAN.K2,
                                       search_type=cfg.PSEUDO.SEARCH_TYPE).cpu().numpy()
    features = features.cpu()
    # clustering
    if cfg.PSEUDO.DBSCAN.BASE == 'rho':
        rho = cfg.PSEUDO.DBSCAN.RHO
        tri_mat = np.triu(dist, 1) # tri_mat.dim=2
        tri_mat = tri_mat[np.nonzero(tri_mat)] # tri_mat.dim=1
        tri_mat = np.sort(tri_mat,axis=None)
        top_num = np.round(rho * tri_mat.size).astype(int)  # rho=3.4e-6, eps is around 0.6
        eps = tri_mat[:top_num].mean()
        if len(cfg.PSEUDO.DBSCAN.EPS) == 3:
            eps = [eps - 0.02, eps, eps + 0.02]
        else:
            eps = [eps]
    else:
        eps = cfg.PSEUDO.DBSCAN.EPS
    logger.info(f'dbscan based on {cfg.PSEUDO.DBSCAN.BASE}, eps in dbscan clustering: {eps}')

    if len(eps) == 1:
        # normal clustering
        labels, centers, num_classes = label_generator_dbscan_single(cfg, features, dist, eps[0])
        return labels, centers, num_classes, indep_thres, torch.Tensor(dist)
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

        return labels_normal, centers, num_classes, indep_thres, torch.Tensor(dist)


@torch.no_grad()
def label_generator_cdp(cfg, features, indep_thres=None, **kwargs):
    k = cfg.PSEUDO.CDP.K
    th = cfg.PSEUDO.CDP.VOT.THRESHOLD
    assert cfg.PSEUDO.CDP.STRATEGY == 'vote'

    if len(th) == 1 and len(k) == 1:
        labels, centers, num_classes = label_generator_cdp_single(cfg, features, k[0], th[0], indep_thres, **kwargs)
        return labels, centers, num_classes, indep_thres, None
    else:
        k = [20, 25, 30][::-1]
        th = [0.35, 0.4, 0.45][::-1]
        labels_tight, _, a = label_generator_cdp_single(cfg, features, k[0], th[0], **kwargs)
        labels_normal, _, num_classes = label_generator_cdp_single(cfg, features, k[1], th[1], **kwargs)
        labels_loose, _, b = label_generator_cdp_single(cfg, features, k[2], th[2], **kwargs)
        print(a, num_classes, b)
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

        return labels_normal, centers, num_classes, indep_thres, None

@torch.no_grad()
def label_generator_cdp_single(cfg, features, k, th, **kwargs):
    # faiss
    index = faiss.IndexFlatL2(features.size(-1))
    index.add(features.cpu().numpy())
    knn_dist, knn = index.search(features.cpu().numpy(), k)
    knn_dist /= 2  # scale to [0, 1]

    max_sz = cfg.PSEUDO.CDP.PROPAGATION.MAX_SIZE
    step = cfg.PSEUDO.CDP.PROPAGATION.STEP
    max_iter = cfg.PSEUDO.CDP.PROPAGATION.MAX_ITER
    use_outliers = cfg.PSEUDO.USE_OUTLIERS

    # vote to build graph
    simi = 1.0 - knn_dist
    anchor = np.tile(np.arange(len(knn)).reshape(len(knn), 1), (1, knn.shape[1]))
    selidx = np.where((simi > th) & (knn != -1) & (knn != anchor))

    pairs = np.hstack((anchor[selidx].reshape(-1, 1), knn[selidx].reshape(-1, 1)))
    scores = simi[selidx]
    pairs = np.sort(pairs, axis=1)
    pairs, unique_idx = np.unique(pairs, return_index=True, axis=0)
    scores = scores[unique_idx]

    # propagation
    components = graph_propagation(pairs, scores, max_sz, step, max_iter)

    # collect results
    cdp_res = []
    for c in components:
        cdp_res.append(sorted([n.name for n in c]))
    pred = -1 * np.ones(features.size(0), dtype=np.int)
    for i,c in enumerate(cdp_res):
        pred[np.array(c)] = i

    valid = np.where(pred != -1)
    _, unique_idx = np.unique(pred[valid], return_index=True)
    pred_unique = pred[valid][np.sort(unique_idx)]
    pred_mapping = dict(zip(list(pred_unique), range(pred_unique.shape[0])))
    pred_mapping[-1] = -1
    pred = np.array([pred_mapping[p] for p in pred])

    # get num_cluster and centers
    num_clusters = len(set(pred)) - (1 if -1 in pred else 0)
    centers = collections.defaultdict(list)
    outliers = 0
    for i, label in enumerate(pred):
        if label == -1:
            if not use_outliers:
                continue
            pred[i] = num_clusters + outliers
            outliers += 1
        centers[pred[i]].append(features[i])

    centers = [torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())]
    centers = torch.stack(centers, dim=0)
    labels = to_torch(pred).long()
    num_clusters += outliers

    return labels, centers, num_clusters


def label_generator_hypergraph(cfg, features, num_classes=500, **kwargs):
    assert isinstance(num_classes, int), 'unsupported num_classes: {}'.format(num_classes)

    use_outliers = cfg.PSEUDO.USE_OUTLIERS
    # construct hypergraph
    feat_np = features.numpy()
    H_list = []
    for k in [20, 30]:
        H = gen_knn_hg(feat_np, n_neighbors=k, is_prob=True, with_feature=False)
        H_list.append(H)
    H = concat_multi_hg(H_list)
    labels = spectral_hg_partitioning(H, n_clusters=num_classes)

    # cluster labels -> pseudo labels
    centers = collections.defaultdict(list)
    outliers = 0
    for i, label in enumerate(labels):
        if label == -1:
            if not use_outliers:
                continue
            labels[i] = num_classes + outliers
            outliers += 1
        centers[labels[i]].append(features[i])

    centers = [torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())]
    centers = torch.stack(centers, dim=0)
    labels = to_torch(labels).long()
    num_classes += outliers

    return labels, centers, num_classes, None, None


def label_generator_hypergraph_dbscan(cfg, features, **kwargs):
    global preH
    # initial clustering
    if 'dist' in kwargs:
        dist = kwargs['dist']
    else:
        dist = compute_distance_matrix(features,
                                       None,
                                       metric=cfg.PSEUDO.DBSCAN.DIST_METRIC,
                                       k1=cfg.PSEUDO.DBSCAN.K1,
                                       k2=cfg.PSEUDO.DBSCAN.K2,
                                       search_type=cfg.PSEUDO.SEARCH_TYPE).cpu().numpy()
    features = features.cpu()
    # clustering
    if cfg.PSEUDO.DBSCAN.BASE == 'rho':
        rho = cfg.PSEUDO.DBSCAN.RHO
        tri_mat = np.triu(dist, 1) # tri_mat.dim=2
        tri_mat = tri_mat[np.nonzero(tri_mat)] # tri_mat.dim=1
        tri_mat = np.sort(tri_mat,axis=None)
        top_num = np.round(rho * tri_mat.size).astype(int)  # rho=3.4e-6, eps is around 0.6
        eps = tri_mat[:top_num].mean()
        eps = [eps]
    else:
        eps = cfg.PSEUDO.DBSCAN.EPS
    logger.info(f'dbscan based on {cfg.PSEUDO.DBSCAN.BASE}, eps in dbscan clustering: {eps}')

    assert len(eps) == 1
    labels, centers, num_classes = label_generator_dbscan_single(cfg, features, dist, eps[0])

    if 'epoch' in kwargs and kwargs['epoch'] < cfg.PSEUDO.HG.START_EPOCH:
        logger.info(f'Disable hypergraph clustering before epoch {cfg.PSEUDO.HG.START_EPOCH}')
        return labels, centers, num_classes, None, None
    # generate knn hypergraph
    H = build_hg(cfg, features, dist, labels.cpu().numpy(), kwargs)
    preH = H
    # clustering
    _, embeddings = eigsh(H.laplacian(), num_classes, which='SM')
    embeddings = torch.from_numpy(embeddings).to(torch.float)
    embeddings = torch.nn.functional.normalize(embeddings)
    dist = compute_distance_matrix(embeddings,
                                    None,
                                    metric=cfg.PSEUDO.DBSCAN.DIST_METRIC,
                                    k1=cfg.PSEUDO.DBSCAN.K1,
                                    k2=cfg.PSEUDO.DBSCAN.K2,
                                    search_type=cfg.PSEUDO.SEARCH_TYPE).cpu().numpy()
    if cfg.PSEUDO.DBSCAN.BASE == 'rho':
        rho = cfg.PSEUDO.DBSCAN.RHO
        tri_mat = np.triu(dist, 1) # tri_mat.dim=2
        tri_mat = tri_mat[np.nonzero(tri_mat)] # tri_mat.dim=1
        tri_mat = np.sort(tri_mat,axis=None)
        top_num = np.round(rho * tri_mat.size).astype(int)  # rho=3.4e-6, eps is around 0.6
        eps = tri_mat[:top_num].mean()
        eps = [eps]
    else:
        eps = cfg.PSEUDO.DBSCAN.EPS
    logger.info(f'dbscan based on {cfg.PSEUDO.DBSCAN.BASE}, eps after hypergraph partitioning in dbscan clustering: {eps}')
    labels, centers, num_classes = label_generator_dbscan_single(cfg, features, dist, eps[0])

    return labels, centers, num_classes, None, None


def label_generator_hypergraph_lp(cfg, features, **kwargs):
    pass


def build_hg(cfg, features, dist_mat, pseudo_labels, kwargs):
    H_list = []
    for kk in cfg.PSEUDO.HG.KNN:  # knn graph
        H_list.append(gen_knn_hg(dist_mat, n_neighbors=kk, is_prob=True))
    if 'imgs_path' in kwargs and cfg.PSEUDO.HG.WITH_ST:  # spatial temporal graph
        imgs_path = kwargs['imgs_path']
        st_time = time.time()
        app_dist_mat = compute_distance_matrix(features,
                                               features,
                                               metric='cosine').cpu().numpy()
        st_dist_mat = get_st_matrix(imgs_path, pseudo_labels=pseudo_labels, score=(1 - app_dist_mat))
        st_dist_mat = re_ranking_dist(st_dist_mat, lambda_value=0.5)
        end_time = time.time()
        logger.info(f'get spatial temporal distribution costs {end_time - st_time}s')
        H_list.append(gen_knn_hg(st_dist_mat, n_neighbors=kk, is_prob=True))
    if cfg.PSEUDO.HG.WITH_CLUSTER:  # graph based on clustering
        H_list.append(gen_clustering_hg(dist_mat, pseudo_labels))
    if preH is not None and cfg.PSEUDO.HG.WITH_PREVIOUS:  # dist_mat from last generation
        H_list.append(preH)
    H = concat_multi_hg(H_list)
    return H


def label_generator_dbscan_with_st(cfg, features, **kwargs):
    # initial clustering
    if 'dist' in kwargs:
        dist = kwargs['dist']
    else:
        dist = compute_distance_matrix(features,
                                       None,
                                       metric=cfg.PSEUDO.DBSCAN.DIST_METRIC,
                                       k1=cfg.PSEUDO.DBSCAN.K1,
                                       k2=cfg.PSEUDO.DBSCAN.K2,
                                       search_type=cfg.PSEUDO.SEARCH_TYPE).cpu().numpy()
    features = features.cpu()
    # clustering
    if cfg.PSEUDO.DBSCAN.BASE == 'rho':
        rho = cfg.PSEUDO.DBSCAN.RHO
        tri_mat = np.triu(dist, 1) # tri_mat.dim=2
        tri_mat = tri_mat[np.nonzero(tri_mat)] # tri_mat.dim=1
        tri_mat = np.sort(tri_mat,axis=None)
        top_num = np.round(rho * tri_mat.size).astype(int)
        eps = tri_mat[:top_num].mean()
        eps = [eps]
    else:
        eps = cfg.PSEUDO.DBSCAN.EPS
    logger.info(f'dbscan based on {cfg.PSEUDO.DBSCAN.BASE}, eps in dbscan clustering: {eps}')

    assert len(eps) == 1
    labels, centers, num_classes = label_generator_dbscan_single(cfg, features, dist, eps[0])
    if 'epoch' in kwargs and kwargs['epoch'] < cfg.PSEUDO.ST.START_EPOCH:
        logger.info(f'Use appearance distance matrix for clustering before epoch {cfg.PSEUDO.ST.START_EPOCH}')
        return labels, centers, num_classes, None, None
    logger.info('Use appearance and spatial temporal information for clustering.')
    assert 'imgs_path' in kwargs, "imgs_path must be in kwargs, while only {}".format(kwargs.keys())
    app_dist_mat = compute_distance_matrix(features,
                                           features,
                                           metric='cosine').cpu().numpy()
    st_time = time.time()
    dist = get_st_matrix(kwargs['imgs_path'], pseudo_labels=labels.tolist(), score=(1 - app_dist_mat))
    dist = re_ranking_dist(dist, lambda_value=0.5)
    logger.info(f'get spatial temporal distance matrix costs {time.time() - st_time}s')
    # clustering based on appearance and spatial temporal information
    if cfg.PSEUDO.DBSCAN.BASE == 'rho':
        rho = cfg.PSEUDO.DBSCAN.RHO
        tri_mat = np.triu(dist, 1) # tri_mat.dim=2
        tri_mat = tri_mat[np.nonzero(tri_mat)] # tri_mat.dim=1
        tri_mat = np.sort(tri_mat,axis=None)
        top_num = np.round(rho * tri_mat.size).astype(int)
        eps = tri_mat[:top_num].mean()
        eps = [eps]
    else:
        eps = cfg.PSEUDO.DBSCAN.EPS
    logger.info(f'dbscan based on {cfg.PSEUDO.DBSCAN.BASE}, eps in dbscan clustering: {eps}')
    assert len(eps) == 1
    labels, centers, num_classes = label_generator_dbscan_single(cfg, features, dist, eps[0])
    return labels, centers, num_classes, None, None


def label_generator_dbscan_with_cams(cfg, features, cams, epoch, **kwargs):
    dist = compute_distance_matrix(features,
                                    None,
                                    metric=cfg.PSEUDO.DBSCAN.DIST_METRIC,
                                    k1=cfg.PSEUDO.DBSCAN.K1,
                                    k2=cfg.PSEUDO.DBSCAN.K2,
                                    search_type=cfg.PSEUDO.SEARCH_TYPE).cpu().numpy()
    if epoch >= cfg.PSEUDO.DBSCAN.CAMERA.EPOCH[0] and epoch <= cfg.PSEUDO.DBSCAN.CAMERA.EPOCH[1]:
        labels = -1 * torch.ones(features.shape[0]).long()
        centers = []
        logger.info('Use Camera-aware clustering on epoch {}'.format(epoch))
        # perform dbscan on every camera
        num_classes = 0
        for cc in torch.unique(cams):
            ind = torch.where(cams == cc)[0]
            features_tmp = features[ind]
            labels_tmp, centers_tmp, num_classes_tmp, _, _ = label_generator_dbscan(cfg, features_tmp, **kwargs)
            inlier_ind = torch.where(labels_tmp >= 0)[0]
            labels_tmp[inlier_ind] += num_classes
            centers.append(centers_tmp)
            num_classes += num_classes_tmp
            labels[ind] = labels_tmp.long()
        centers = torch.cat(centers, dim=0)
        return labels, centers, num_classes, None, torch.tensor(dist)
    else:
        logger.info('Use global clustering on epoch {}'.format(epoch))
        return label_generator_dbscan(cfg, features, dist=dist, **kwargs)
