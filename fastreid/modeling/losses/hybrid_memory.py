import torch
import torch.nn.functional as F
from torch import nn, autograd
from fastreid.utils.comm import all_gather_tensor, all_gather, get_world_size, concat_all_gather
from fastreid.utils.metrics import compute_distance_matrix


class HM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        outputs = inputs.mm(ctx.features.t())
        if get_world_size() > 1:  # distributed
            all_inputs = all_gather_tensor(inputs)
            all_indexes = all_gather_tensor(indexes)
        else:  # error when use DDP
            all_inputs = torch.cat(all_gather(inputs), dim=0)
            all_indexes = torch.cat(all_gather(indexes), dim=0)
        ctx.save_for_backward(all_inputs, all_indexes)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def hm(inputs, indexes, features, momentum=0.5):
    return HM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class HybridMemory(nn.Module):
    def __init__(self, num_features, num_memory, temp=0.05, momentum=0.2, weighted=True, weight_mask_topk=-1, soft_label=True, soft_label_start_epoch=20):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_memory = num_memory
        self.weighted = weighted
        self.weight_mask_topk = weight_mask_topk
        self.soft_label = soft_label
        self.soft_label_start_epoch = soft_label_start_epoch
        self.cur_epoch = 0

        self.momentum = momentum
        self.temp = temp

        self.register_buffer("features", torch.zeros(num_memory, num_features))
        self.register_buffer("labels", torch.zeros(num_memory).long())
        self.centers = None
        self.weight_matrix = None

    def _update_epoch(self, epoch):
        self.cur_epoch = epoch

    @torch.no_grad()
    def _update_center(self, centers):
        centers = F.normalize(centers, p=2, dim=1)
        self.centers = centers.float().to(self.features.device)

    @torch.no_grad()
    def _update_feature(self, features):
        features = F.normalize(features, p=2, dim=1)
        self.features.data.copy_(features.float().to(self.features.device))

    @torch.no_grad()
    def _update_weight_matrix(self, dist_mat, labels):
        num_classes = len(torch.unique(labels))
        if -1 in labels:
            num_classes -= 1
        weight_matrix = torch.zeros((len(labels), num_classes))
        nums = torch.zeros((1, num_classes))
        index_select = torch.where(labels >= 0)[0]
        inputs_select = dist_mat[index_select].t()
        labels_select = labels[index_select]
        weight_matrix.index_add_(1, labels_select, inputs_select)
        nums.index_add_(1, labels_select, torch.ones(1, len(index_select)))
        weight_matrix = 1 - weight_matrix / nums
        self.weight_matrix = weight_matrix

    @torch.no_grad()
    def _update_label(self, labels):
        self.labels.data.copy_(labels.long().to(self.labels.device))

    @torch.no_grad()
    def _pseudo_label(self, indexes, tau=30):
        # pseudo label current generation
        pseudo_label = 1 - compute_distance_matrix(self.features[indexes].detach(), self.centers.detach(), metric='cosine').cuda()
        pseudo_label = F.softmax(pseudo_label * tau, dim=1)
        return pseudo_label

    def forward(self, inputs, indexes, eps=1e-6, **kwargs):
        inputs = F.normalize(inputs, p=2, dim=1)
        indexes = indexes.cuda()
        # inputs: B*2048, features: L*2048
        inputs = hm(inputs, indexes, self.features, self.momentum)
        inputs /= self.temp
        B = inputs.size(0)

        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return masked_exps / masked_sums

        targets = self.labels[indexes].clone()
        labels = self.labels.clone()

        if labels.min() >= 0:
            sim = torch.zeros(B, labels.max() + 1).float().cuda()
            sim.index_add_(1, labels, inputs)
            nums = torch.zeros(1, labels.max() + 1).float().cuda()
            nums.index_add_(1, labels, torch.ones(1, self.num_memory).float().cuda())
        else:
            sim = torch.zeros(B, labels.max() + 1).float().cuda()
            nums = torch.zeros(1, labels.max() + 1).float().cuda()
            index_select = torch.where(labels >= 0)[0]
            inputs_select = inputs.t().contiguous()[index_select]
            labels_select = labels[index_select]
            sim.index_add_(1, labels_select, inputs_select)
            nums.index_add_(1, labels_select, torch.ones(1, len(index_select)).float().cuda())
        mask = (nums > 0).float()
        sim /= (mask * nums + (1 - mask)).clone().expand_as(sim)
        mask = mask.expand_as(sim)
        masked_sim = masked_softmax(sim, mask)
        if self.weighted and self.weight_matrix is not None:
            weight = self.weight_matrix[indexes].cuda().clone() / self.temp
            if self.weight_mask_topk > 0:
                mask_index = weight.argsort(dim=1, descending=True)[:, :self.weight_mask_topk]
                weight_mask = torch.zeros_like(weight)
                weight_mask.scatter_(dim=1, index=mask_index, src=torch.ones_like(weight))
                weight = masked_softmax(weight, weight_mask)
            loss = -(torch.log(masked_sim + eps) * F.softmax(weight, 1)).mean(0).sum()
        elif self.soft_label and self.cur_epoch > self.soft_label_start_epoch:
            pseudo_label = self._pseudo_label(indexes)
            loss = -(torch.log(masked_sim + eps) * pseudo_label).mean(0).sum()
        else:
            loss = F.nll_loss(torch.log(masked_sim + eps), targets)
        return {'contrastive_loss': loss}
