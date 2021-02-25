from collections import defaultdict
import torch
from torch import nn, autograd
import torch.nn.functional as F

from fastreid.utils.comm import all_gather_tensor, all_gather, get_world_size

class Proxy(autograd.Function):
    @staticmethod
    def forward(ctx, feats, indexes, centers, momentum):
        ctx.centers = centers
        ctx.momentum = momentum
        outputs = feats
        if get_world_size() > 1:  # distributed
            all_feats = all_gather_tensor(feats)
            all_indexes = all_gather_tensor(indexes)
        else:  # error when use DDP
            all_feats = torch.cat(all_gather(feats), dim=0)
            all_indexes = torch.cat(all_gather(indexes), dim=0)
        ctx.save_for_backward(all_feats, all_indexes)

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        feats, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs

        # momentum update
        for x, y in zip(feats, indexes):
            ctx.centers[y] = ctx.momentum * ctx.centers[y]  + (1.0 - ctx.momentum) * x
            ctx.centers[y] /= ctx.centers[y].norm() 

        return grad_inputs, None, None, None, None, None


class CAPMemory(nn.Module):
    def __init__(self, cfg):
        super(CAPMemory, self).__init__()
        self.cfg = cfg
        self.t = self.cfg.CAP.TEMP # temperature
        self.hard_neg_k = self.cfg.CAP.HARD_NEG_K # hard negative sampling in global loss
        self.momentum = self.cfg.CAP.MOMENTUM # momentum update rate
        self.intercam_epoch = self.cfg.CAP.INTERCAM_EPOCH
        self.cur_epoch = 0

    def _update_epoch(self, epoch):
        self.cur_epoch = epoch

    def _update_center(self, centers, labels, camids):
        self.labels = torch.tensor(labels).cuda()
        self.cams = camids.clone().detach().cuda()

        self.centers = []
        self.camlabel2proxy = defaultdict(int)
        self.cam2proxy = defaultdict(list)
        self.label2proxy = defaultdict(list)
        for i, (label, camid) in enumerate(centers):
            self.centers.append(centers[label, camid])
            self.camlabel2proxy[(label, camid)] = i
            self.cam2proxy[camid].append(i)
            self.label2proxy[label].append(i)
        self.centers = torch.cat(self.centers).cuda()
        self.temp_centers = self.centers.detach().clone()

    def forward(self, feats, indexes, **kwargs):
        if self.cfg.CAP.NORM_FEAT:
            feats = F.normalize(feats, p=2, dim=1)

        labels = self.labels[indexes]
        cams = self.cams[indexes]
        n = feats.size(0)
        # update memory centers
        mapped_indexes = torch.tensor([self.camlabel2proxy[(labels[i].item(), cams[i].item())] for i in range(n)]).cuda()
        feats = Proxy.apply(feats, mapped_indexes, self.centers, torch.Tensor([self.momentum]).to(feats.device))
        loss_intra_list = []
        loss_inter_list = []
        for i in range(n):
            x = feats[i].unsqueeze(0)
            proxy_ind = mapped_indexes[i].item()
            # intra loss
            intra_ind = self.cam2proxy[cams[i].item()]
            assert proxy_ind in intra_ind
            intra_centers = self.centers[intra_ind]
            loss_intra_list.append(-F.log_softmax(x.mm(intra_centers.t()) / self.t, dim=1)[:, intra_ind.index(proxy_ind)])
            # inter loss
            if self.cur_epoch >= self.intercam_epoch:
                inter_pos_ind = self.label2proxy[labels[i].item()]
                inter_all_sims = x.mm(self.temp_centers.t().clone()).squeeze(0)
                temp_all_sims = inter_all_sims.detach().clone()
                inter_all_sims /= self.t
                temp_all_sims[inter_pos_ind] = -10000.0 # mask positive
                inter_neg_ind = torch.sort(temp_all_sims)[1][-self.hard_neg_k:]
                concated_input = torch.cat((inter_all_sims[inter_pos_ind], inter_all_sims[inter_neg_ind]), dim=0)
                concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).cuda()
                concated_target[0:len(inter_pos_ind)] = 1.0 / len(inter_pos_ind)
                loss_inter_list.append((-F.log_softmax(concated_input, dim=0) * concated_target).sum())

        loss_intra_list = torch.cat(loss_intra_list)
        if self.cur_epoch >= self.intercam_epoch:
            loss_inter_list = torch.stack(loss_inter_list)
        loss_intra = torch.Tensor([0]).cuda()
        loss_inter = torch.Tensor([0]).cuda()
        for cc in torch.unique(cams):
            inds = torch.where(cams == cc)[0]
            loss_intra += loss_intra_list[inds].mean()
            if self.cur_epoch >= self.intercam_epoch:
                loss_inter += loss_inter_list[inds].mean()

        if self.cur_epoch < self.intercam_epoch:
            return {"loss_intra": loss_intra}
        else:
            return {"loss_intra": loss_intra, "loss_inter": self.cfg.CAP.LOSS_WEIGHT * loss_inter}
