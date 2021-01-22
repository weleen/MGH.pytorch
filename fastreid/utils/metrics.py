import faiss
import torch
import torch.nn.functional as F
import time
import logging
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

import build_adjacency_matrix
import gnn_propagate

from .faiss_utils import (
    index_init_cpu,
    index_init_gpu,
    search_index_pytorch,
    search_raw_array_pytorch,
)

logger = logging.getLogger(__name__)

def compute_distance_matrix(input1, input2, metric='euclidean', **kwargs) -> torch.Tensor:
    """A wrapper function for computing distance matrix.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
        metric (str, optional): "euclidean", "cosine", "jaccard" or "hamming".
            Default is "euclidean".

    Returns:
        torch.Tensor: distance matrix.

    Examples::
       >>> from utils import metrics
       >>> input1 = torch.rand(10, 2048)
       >>> input2 = torch.rand(100, 2048)
       >>> distmat = metrics.compute_distance_matrix(input1, input2)
       >>> distmat.size() # (10, 100)
    """
    # check input
    assert isinstance(input1, torch.Tensor)
    assert input1.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input1.dim()
    )
    if input2 is not None: 
        assert isinstance(input2, torch.Tensor)
        assert input2.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
            input2.dim()
        )
        assert input1.size(1) == input2.size(1)

    if metric == 'euclidean':
        distmat = euclidean_dist(input1, input2)
    elif metric == 'cosine':
        distmat = cosine_dist(input1, input2)
    elif metric == 'hamming':
        distmat = hamming_distance(input1, input2)
    elif metric == 'jaccard':
        if input2 is not None:
            feat = torch.cat((input1, input2), dim=0)
            distmat = jaccard_dist(feat, **kwargs)
            distmat = distmat[:input1.size(0), input1.size(0):]
        else:
            distmat = jaccard_dist(input1, **kwargs)
    elif metric == 'gnn':
        if input2 is None:
            distmat = gnn_dist(input1, input1, **kwargs)
        else:
            distmat = gnn_dist(input1, input2, **kwargs)
    else:
        raise ValueError('Unknown distance metric: {}. '
                         'Please choose metric from [euclidean | cosine | hamming | jaccard | gnn]'.format(metric)
        )
        

    return distmat

@torch.no_grad()
def gnn_dist(X_q, X_g, k1=26, k2=8, **kwargs):
    # X_q and X_g should be normalized
    X_q = X_q.cuda()
    X_g = X_g.cuda()
    query_num = X_q.shape[0]

    X_u = torch.cat((X_q, X_g), axis=0)
    original_score = torch.mm(X_u, X_u.t())
    del X_u, X_q, X_g

    # initial ranking list
    S, initial_rank = original_score.topk(k=k1, dim=-1, largest=True, sorted=True)
    
    # stage 1
    A = build_adjacency_matrix.forward(initial_rank.float())
    S = S * S

    # stage 2
    if k2 != 1:
        for _ in range(2):
            A = A + A.T
            A = gnn_propagate.forward(A, initial_rank[:, :k2].contiguous().float(), S[:, :k2].contiguous().float())
            A_norm = torch.norm(A, p=2, dim=1, keepdim=True)
            A = A.div(A_norm.expand_as(A))

    dist = cosine_dist(A[:query_num,], A[query_num:, ]).cpu()
    return dist


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]


def jaccard_dist(features, k1=20, k2=6, search_option=0, fp16=False, **kwargs):
    start = time.time()
    ngpus = faiss.get_num_gpus()
    N = features.size(0)
    mat_type = np.float16 if fp16 else np.float32

    if search_option == 0:
        # GPU + PyTorch CUDA Tensors (1)
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        _, initial_rank = search_raw_array_pytorch(res, features, features, k1)
        initial_rank = initial_rank.cpu().numpy()
    elif search_option == 1:
        # GPU + PyTorch CUDA Tensors (2)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, features.size(-1))
        index.add(features.cpu().numpy())
        _, initial_rank = search_index_pytorch(index, features, k1)
        res.syncDefaultStreamCurrentDevice()
        initial_rank = initial_rank.cpu().numpy()
    elif search_option == 2:
        # GPU
        index = index_init_gpu(ngpus, features.size(-1))
        index.add(features.cpu().numpy())
        _, initial_rank = index.search(features.cpu().numpy(), k1)
    else:
        # CPU
        index = index_init_cpu(features.size(-1))
        index.add(features.cpu().numpy())
        _, initial_rank = index.search(features.cpu().numpy(), k1)

    nn_k1 = []
    nn_k1_half = []
    for i in range(N):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1 / 2))))

    V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(
            k_reciprocal_expansion_index)  # element-wise unique

        x = features[i].unsqueeze(0).contiguous()
        y = features[k_reciprocal_expansion_index]
        m, n = x.size(0), y.size(0)
        dist = (
            torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n)
            + torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        )
        dist.addmm_(x, y.t(), beta=1, alpha=-2)

        if fp16:
            V[i, k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(mat_type)
        else:
            V[i, k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()

    del nn_k1, nn_k1_half, x, y
    features = features.cpu()

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=mat_type)
        for i in range(N):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe

    del initial_rank

    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:, i] != 0)[0])  # len(invIndex)=all_num

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=mat_type)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]]
            )

        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    del invIndex, V

    pos_bool = jaccard_dist < 0
    jaccard_dist[pos_bool] = 0.0
    logger.info("Jaccard distance computing time cost: {}".format(time.time() - start))

    return torch.Tensor(jaccard_dist)


@torch.no_grad()
def euclidean_dist(x, y):
    """Computes euclidean distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


@torch.no_grad()
def cosine_dist(input1, input2):
    """Computes cosine distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    return distmat.clamp(min=0, max=2)


@torch.no_grad()
def hamming_distance(input1, input2):
    """Computes hamming distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix. {0, 1}.
        input2 (torch.Tensor): 2-D feature matrix. {0, 1}.

    Returns:
        torch.Tensor: distance matrix.
    """
    input1 = input1.to(torch.int)
    input2 = input2.to(torch.int)
    input1_m1 = input1 - 1
    input2_m1 = input2 - 1
    c1 = input1.matmul(input2_m1.T)
    c2 = input1_m1.matmul(input2.T)
    return torch.abs(c1 + c2)


def cluster_accuracy(output, target):
    """
    Calculate clustering accuracy.
    :param output: (numpy.array): predicted matrix with shape (batch_size,)
    :param target: (numpy.array): ground truth with shape (batch_size,)
    :return:
    """
    target = target.astype(np.int64)
    assert output.size == target.size
    D = max(output.max(), target.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(output.size):
        w[output[i], target[i]] += 1
    row_ind, col_ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / output.size


def purity(output, target, min_samples=2):
    """Computes custom clustering metric, for each cluster, if all instances in this cluster has  the same ground truth label, correct_cnt add 1, ignore outlier with label=-1. When the number of samples in a cluster is lower than min_samples, these samples are ignored in calculation.
    :param output: (numpy.array): predicted matrix with shape (batch_size,)
    :param target: (numpy.array): ground truth with shape (batch_size,)
    """
    correct_cnt = 0
    all_cnt = 0
    pid_set = np.unique(output).tolist()
    if -1 in pid_set:
        pid_set.remove(-1)
    for pid in pid_set:
        cluster_index = np.where(output == pid)[0]
        selected_gt_label = target[cluster_index]
        if len(selected_gt_label) < min_samples:
            continue
        if len(np.unique(selected_gt_label)) == 1:
            correct_cnt += 1
        all_cnt += 1
    return correct_cnt / (all_cnt + 1e-6)


def cluster_metrics(label_pred: np.ndarray, label_true: np.ndarray):
    """
    Calculate clustering accuracy, nmi and ari
    :param label_pred(np.int64): predicted matrix with shape (N,)
    :param label_true(np.int64): ground truth with shape (N,)
    :return: Tuple(float, float, float) for acc, nmi, ari
    """
    assert label_true.dtype == label_pred.dtype == np.int64, "dtype error."
    nmi_score = normalized_mutual_info_score(label_true, label_pred)
    ari_score = adjusted_rand_score(label_true, label_pred)
    # acc_score = cluster_accuracy(label_pred, label_true)
    purity_score = purity(label_pred, label_true)
    return nmi_score, ari_score, purity_score #, acc_score
