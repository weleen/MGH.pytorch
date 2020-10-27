import torch
import torch.nn.functional as F
from torch import nn, autograd
from fastreid.utils.comm import all_gather_tensor, all_gather
from fastreid.utils.metrics import compute_distance_matrix


class HM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        outputs = inputs.mm(ctx.features.t())
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
    def __init__(self, num_features, num_memory, temp=0.05, momentum=0.2, weight_mask_topk=3):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_memory = num_memory

        self.momentum = momentum
        self.temp = temp
        self.weight_mask_topk = weight_mask_topk

        self.register_buffer("features", torch.zeros(num_memory, num_features))
        self.register_buffer("labels", torch.zeros(num_memory).long())

    @torch.no_grad()
    def _update_feature(self, features):
        features = F.normalize(features, p=2, dim=1)
        self.features.data.copy_(features.float().to(self.features.device))

    @torch.no_grad()
    def _update_label(self, labels):
        self.labels.data.copy_(labels.long().to(self.labels.device))

    def forward(self, inputs, indexes, weight=None, **kwargs):
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

        sim = torch.zeros(B, labels.max() + 1).float().cuda()
        sim.index_add_(1, labels, inputs)
        nums = torch.zeros(1, labels.max() + 1).float().cuda()
        nums.index_add_(1, labels, torch.ones(1, self.num_memory).float().cuda())
        mask = (nums > 0).float()
        sim /= (mask * nums + (1 - mask)).clone().expand_as(sim)
        mask = mask.expand_as(sim)
        masked_sim = masked_softmax(sim, mask)
        if weight is None:
            return F.nll_loss(torch.log(masked_sim + 1e-6), targets)
        else:
            weight = weight[indexes].cuda() / self.temp
            # 1. F.softmax
            # return -(torch.log(masked_sim + 1e-6) * F.softmax(weight, 1)).mean(0).sum()
            # 2. use mask
            # weight_mask = (weight > 0.01 / self.temp).float()
            # 3. select topk clusters as positive clusters
            mask_index = weight.argsort(dim=1, descending=True)[:, :self.weight_mask_topk]
            weight_mask = torch.zeros_like(weight)
            weight_mask.scatter_(dim=1, index=mask_index, src=torch.ones_like(weight))
            masked_weight = masked_softmax(weight, weight_mask)
            loss = -(torch.log(masked_sim + 1e-6) * masked_weight).mean(0).sum()
            if loss > 1e3 or loss < 1e-3:
                print('debug')
            return loss

    def circle_loss(self, sim, label_matrix=None, type='class'):
        if type =='class':
            # class-wise label circle loss
            pass
        elif type == 'pair':
            # pair-wise label circle loss
            assert label_matrix is not None
            pass
        else:
            raise NameError(f'{type} is not supported, please select from class and pair.')
