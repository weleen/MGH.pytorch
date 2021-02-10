# coding=utf-8
import faiss
import torch
import time
import logging
import numpy as np
from sklearn.cluster import k_means, DBSCAN
from sklearn.metrics import pairwise_distances

from scipy.linalg import eigh
from scipy.sparse.linalg.eigen.arpack import eigsh
from fastreid.utils.torch_utils import to_numpy, to_torch
from fastreid.utils.metrics import compute_distance_matrix

from hyperg.hyperg import HyperG

logger = logging.getLogger('fastreid.' + __name__)

def spectral_hg_partitioning(hg, n_clusters, cluster_method='dbscan', n_components=None):
    """
    :param hg: instance of HyperG
    :param n_clusters: int,
    :param assign_labels: str, {'kmeans', 'discretize'}, default: 'kmeans'
    :param n_components: int, number of eigen vectors to use for the spectral embedding
    :param random_state: int or None (default)
    :param n_init: int, number of time the k-means algorithm will be run
    with different centroid seeds.
    :return: numpy array, shape = (n_samples,), labels of each point
    """

    assert isinstance(hg, HyperG)
    assert n_clusters <= hg.num_nodes()

    if n_components is None:
        n_components = n_clusters

    st = time.time()
    L = hg.laplacian().toarray()
    # L = check_symmetric(L)
    st1 = time.time()

    # eigenval, eigenvec = eigh(L)
    # embeddings = eigenvec[:, :n_components]
    # eigenval_1, embeddings = eigh(L, eigvals=(0, n_clusters-1))
    _, embeddings = eigsh(hg.laplacian(), n_clusters, which='SM')
    st2 = time.time()
    print(f'eigen time: {st2 - st1}')

    if cluster_method == 'kmeans':
        _, labels, _ = k_means(embeddings, n_clusters)
    elif cluster_method == 'dbscan':
        embeddings = torch.from_numpy(embeddings).to(torch.float)
        embeddings = torch.nn.functional.normalize(embeddings)
        st = time.time()
        dist_mat_jaccard = compute_distance_matrix(embeddings, None, metric='jaccard', k1=30).cpu().numpy()
        print('calculate jaccard distance: {}'.format(time.time() - st))

        eps=0.6
        dbscan_cluster = DBSCAN(eps=eps, min_samples=4, metric="precomputed", n_jobs=-1)
        labels = dbscan_cluster.fit_predict(dist_mat_jaccard)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print('num_cluster: {}'.format(num_clusters))
    st3 = time.time()
    print(f'cluster time: {st3 - st2}')

    return labels.astype(int)
