import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


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
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


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
    return distmat


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


def compute_distance_matrix(input1, input2, metric='euclidean'):
    """A wrapper function for computing distance matrix.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
        metric (str, optional): "euclidean", "cosine" or "hamming".
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
    assert isinstance(input2, torch.Tensor)
    assert input1.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input1.dim()
    )
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
    else:
        raise ValueError(
            'Unknown distance metric: {}. '
            'Please choose either "euclidean" or "cosine"'.format(metric)
        )

    return distmat


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
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / output.size


def cluster_metrics(label_pred: np.ndarray, label_true: np.ndarray):
    """
    Calculate clustering accuracy, nmi and ari
    :param label_pred: predicted matrix with shape (N,)
    :param label_true: ground truth with shape (N,)
    :return: Tuple(float, float, float) for acc, nmi, ari
    """
    nmi_score = normalized_mutual_info_score(label_true, label_pred)
    ari_score = adjusted_rand_score(label_true, label_pred)
    acc_score = cluster_accuracy(label_pred, label_true)
    return acc_score, nmi_score, ari_score
