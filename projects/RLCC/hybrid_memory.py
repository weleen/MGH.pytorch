import os

import torch
import torch.nn.functional as F

from fastreid.modeling.losses.hybrid_memory import HybridMemory
from fastreid.utils.metrics import compute_distance_matrix


class RLCCMemory(HybridMemory):
    def __init__(self, num_features, num_memory, temp=0.05, momentum=0.2, weighted=True, weight_mask_topk=-1,
                 soft_label=True, soft_label_start_epoch=0, rlcc_start_epoch=0):
        super(RLCCMemory, self).__init__(num_features, num_memory, temp, momentum, weighted, weight_mask_topk,
                                         soft_label, soft_label_start_epoch)

        self.rlcc_start_epoch = rlcc_start_epoch

        self.register_buffer("last_features", torch.zeros(num_memory, num_features))
        self.register_buffer("last_labels", torch.zeros(num_memory).long())
        self.last_centers = None

    @torch.no_grad()
    def _update_last_center(self, centers):
        centers = F.normalize(centers, p=2, dim=1)
        self.last_centers = centers.float().to(self.features.device)

    @torch.no_grad()
    def _update_last_feature(self, features):
        features = F.normalize(features, p=2, dim=1)
        self.last_features.data.copy_(features.float().to(self.last_features.device))

    @torch.no_grad()
    def _update_last_label(self, labels):
        self.last_labels.data.copy_(labels.long().to(self.last_labels.device))

    def calculate_mapping_matrix(self, filename):
        # TODO: use cuda to speed up this calculation
        # save last_labels and labels
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save({'last_label': self.last_labels.cpu(), 'label': self.labels.cpu()}, filename)
        res = os.system('python projects/RLCC/calculate_mapping_matrix.py --file {}'.format(filename))
        assert res == 0, 'Run calculate_mapping_matrix.py return {}'.format(res)
        self.C = torch.load(os.path.join(os.path.dirname(filename), 'C.pt')).to(torch.float32).cuda()

    @torch.no_grad()
    def _pseudo_label(self, indexes, alpha=0.9, tau=30):
        # pseudo label current generation (t)
        pseudo_label_2 = 1 - compute_distance_matrix(self.features[indexes].detach(), self.centers.detach(),
                                                     metric='cosine').cuda()
        pseudo_label_2 = F.softmax(pseudo_label_2 * tau, dim=1)

        if self.cur_epoch >= self.rlcc_start_epoch:
            # pseudo label last generation (t-1)
            pseudo_label_1 = 1 - compute_distance_matrix(self.last_features[indexes].detach(),
                                                         self.last_centers.detach(), metric='cosine').cuda()
            pseudo_label_1 = F.softmax(pseudo_label_1 * tau, dim=1)
            # pseudo label propagation 
            propagated_pseudo_label = torch.mm(pseudo_label_1, self.C)
            pseudo_label = alpha * pseudo_label_2 + (1 - alpha) * propagated_pseudo_label
        else:
            pseudo_label = pseudo_label_2

        return pseudo_label
