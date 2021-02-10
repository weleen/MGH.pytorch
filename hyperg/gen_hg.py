# coding=utf-8
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import scipy.sparse as sparse

from hyperg.hyperg import HyperG


def gen_knn_hg(dist_mat, n_neighbors, is_prob=True):
    """
    :param n_neighbors: int,
    :param is_prob: bool, optional(default=True)
    :return: instance of HyperG
    """
    n_nodes, n_edges = dist_mat.shape
    m_neighbors = np.argpartition(dist_mat, kth=n_neighbors+1, axis=1)
    m_neighbors_val = np.take_along_axis(dist_mat, m_neighbors, axis=1)

    m_neighbors = m_neighbors[:, :n_neighbors+1]
    m_neighbors_val = m_neighbors_val[:, :n_neighbors+1]

    for i in range(n_nodes):
        if not np.any(m_neighbors[i, :] == i):
            m_neighbors[i, -1] = i
            m_neighbors_val[i, -1] = 0.

    node_idx = m_neighbors.reshape(-1)
    edge_idx = np.tile(np.arange(n_edges).reshape(-1, 1), (1, n_neighbors+1)).reshape(-1)

    if not is_prob:
        values = np.ones(node_idx.shape[0])
    else:
        avg_dist = np.mean(dist_mat)
        m_neighbors_val = m_neighbors_val.reshape(-1)
        values = np.exp(-np.power(m_neighbors_val, 2.) / np.power(avg_dist, 2.))

    H = sparse.coo_matrix((values, (node_idx, edge_idx)), shape=(n_nodes, n_edges))
    # w = np.ones(n_edges)
    w = np.array(H.sum(axis=0))

    return HyperG(H, w=w)


def gen_clustering_hg(dist_mat, labels, is_prob=True):
    """
    :param X: numpy array, shape = (n_samples, n_features)
    :param n_clusters: int, number of clusters
    :param method: str, clustering methods("dbscan")
    :param with_feature: bool, optional(default=False)
    :param random_state: int, optional(default=False) determines random number generation
    for centroid initialization
    :return: instance of HyperG
    """
    cluster = labels.copy()
    num_clusters = len(set(cluster)) - (1 if -1 in cluster else 0)
    print('num_cluster: {}'.format(num_clusters))
    outliers = 0
    for i, label in enumerate(cluster):
        if label == -1:
            cluster[i] = num_clusters + outliers
            outliers += 1
    n_clusters = num_clusters + outliers

    assert n_clusters >= 1

    n_edges = n_clusters
    n_nodes = dist_mat.shape[0]

    avg_dist = dist_mat.mean()
    node_idx = np.arange(n_nodes)
    edge_idx = cluster

    if is_prob:
        values = []
        for idx in node_idx:
            cluster_idx = np.where(cluster == cluster[idx])[0]
            v = np.mean(dist_mat[idx, cluster_idx])
            values.append(np.exp(-np.power(v, 2.) / np.power(avg_dist, 2.)))
        values = np.array(values)
    else:
        values = np.ones(node_idx.shape[0])
    H = sparse.coo_matrix((values, (node_idx, edge_idx)), shape=(n_nodes, n_edges))
    # w = np.ones(n_edges)
    w = np.array(H.sum(axis=0))

    return HyperG(H, w=w)


def concat_multi_hg(hg_list):
    """concatenate multiple hypergraphs to one hypergraph
    :param hg_list: list, list of HyperG instance
    :return: instance of HyperG
    """
    H_s = [hg.incident_matrix() for hg in hg_list]
    w_s = [hg.hyperedge_weights() for hg in hg_list]

    H = sparse.hstack(H_s)
    w = np.hstack(w_s)

    return HyperG(H, w=w)
