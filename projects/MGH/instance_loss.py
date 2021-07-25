import torch
from fastreid.utils import euclidean_dist

def instance_loss(outs):
    '''
    loss = ||x - x'||_2
    '''
    feats = outs['outputs']['features']
    targets = outs['targets']

    n = targets.size(0)
    mask = targets.expand(n, n).eq(targets.expand(n, n).t())
    mask = torch.tril(mask, diagonal=-1)

    dists = euclidean_dist(feats, feats)
    loss = dists[mask].mean()

    return {'instance_loss': loss}
