import torch
import torch.nn.functional as F

class ActiveTripletLoss(object):
    def __init__(self, cfg):
        # TODO: add in default config
        self._margin = 0.3
        self._scale = 1.0

    def __call__(self, _, global_features, targets):
        _, dim = global_features.size()
        global_features = global_features.view(-1, 3, dim)

        anchors = global_features[:, 0]
        positive = global_features[:, 1]
        negative = global_features[:, 2]

        loss = F.triplet_margin_loss(anchors, positive, negative, margin=self._margin)

        return loss * self._scale
