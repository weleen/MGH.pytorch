import torch
from torch import nn, autograd
import torch.nn.functional as F

from collections import defaultdict


class Proxy(autograd.Function):
    @staticmethod
    def forward(ctx, feats, labels, camids, centers, indexes, momentum):
        ctx.centers = centers
        ctx.indexes = indexes
        ctx.momentum = momentum
        outputs = feats
        ctx.save_for_backward(feats, labels, camids)

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        feats, labels, camids = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs
        
        # momentum update
        for i, feat in enumerate(feats):
            ctx.centers[ctx.indexes[(labels[i], camids[i])]] = ctx.momentum * ctx.centers[ctx.indexes[(labels[i], camids[i])]]  + (1.0 - ctx.momentum) * feat
            # norm centers
            ctx.centers[ctx.indexes[(labels[i], camids[i])]] /= ctx.centers[ctx.indexes[(labels[i], camids[i])]].norm() 

        return grad_inputs, None, None, None, None, None


class CAPMemory(nn.Module):
    def __init__(self, num_features):
        super(CAPMemory, self).__init__()
        self.centers = None
        self.centers_cam = None
        self.centers_lb = None

    def _update_center(self, centers):
        self.centers = []
        self.cam_label_index =defaultdict(int)
        self.cam_index = defaultdict(list)
        self.label_index = defaultdict(list)
        for i, (label, camid) in enumerate(centers):
            self.centers.append(centers[label, camid])
            self.cam_label_index[(label, camid)] = i
            self.cam_index[camid].append(i)
            self.label_index[label].append(i)
        self.centers = torch.cat(self.centers).cuda()
        
    def forward(self, outs, inputs, epoch):
        feats = outs["outputs"]["features"]
        feats = F.normalize(feats, p=2, dim=1)
        labels = inputs["targets"]
        camids = inputs["camids"]
        # inputs p centers * k images
        n = feats.size(0)
        t = 0.07 # temperature
        hard_neg_k = 50
        momentum = 0.2
        lamda = 0.5
        # update memory centers
        feats = Proxy.apply(feats, labels, camids, self.centers, self.cam_label_index, torch.Tensor([momentum]).to(feats.device))

        # intra loss
        loss = []
        for i in range(n):
            x = feats[i]
            x_center = self.centers[self.cam_label_index[(labels[i].item(), camids[i].item())]]
            frac_up = torch.exp((x * x_center).sum() / t)
            x_centers = self.centers[self.cam_index[camids[i].item()]]
            frac_down = torch.exp((x.unsqueeze(0) * x_centers).sum(1) / t).sum() 
            loss.append(torch.log(frac_up / frac_down))
        d = defaultdict(list)
        for i in range(n):
            d[camids[i]].append(loss[i])
        loss_intra = -torch.stack([torch.stack(d[k]).mean() for k in d]).sum()

        if epoch < 5:
            return {"loss_intra": loss_intra}

        # inter loss
        loss = []
        for i in range(n):
            x = feats[i]
            x_center = self.centers[self.cam_label_index[(labels[i].item(), camids[i].item())]]
            frac_up = torch.exp((x * x_center).sum() / t)
            x_centers = self.centers[self.label_index[labels[i].item()]]
            # hard negative
            other_centers = torch.cat([self.centers[self.label_index[lb]] for lb in self.label_index if lb != labels[i]])
            hard_neg = torch.exp((x.unsqueeze(0) * other_centers).sum(1) / t)[:hard_neg_k].sum()
            frac_down = torch.exp((x.unsqueeze(0) * x_centers).sum(1) / t).sum() + hard_neg
            loss.append(torch.log(frac_up / frac_down))
        d = defaultdict(list)
        for i in range(n):
            d[labels[i]].append(loss[i])
        loss_inter = -torch.stack([torch.stack(d[k]).mean() for k in d]).sum()

        return {"loss_intra": loss_intra, "loss_inter": lamda * loss_inter}