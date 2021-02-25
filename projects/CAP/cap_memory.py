import torch
from torch import nn, autograd
import torch.nn.functional as F

from fastreid.utils.comm import all_gather_tensor, all_gather, get_world_size


class ExemplarMemory(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        outputs = inputs.mm(ctx.features.t())
        # if get_world_size() > 1:  # distributed
        #     all_inputs = all_gather_tensor(inputs)
        #     all_indexes = all_gather_tensor(indexes)
        # else:  # error when use DDP
        #     all_inputs = torch.cat(all_gather(inputs), dim=0)
        #     all_indexes = torch.cat(all_gather(indexes), dim=0)
        ctx.save_for_backward(inputs, indexes)
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

    @torch.no_grad()
    def _update_centers_and_labels(self, all_features: list, all_labels: list, all_camids: list):
        assert len(all_features) == len(all_labels) == 1, "Support only one dataset."
        self.cams = torch.cat(all_camids).cuda()
        self.unique_cams = torch.unique(self.cams)
        self.labels = torch.tensor(all_labels[0]).cuda()
        all_features = all_features[0]

        self.percam_memory = []
        self.memory_class_mapper = []
        self.concate_intra_class = []
        for cc in self.unique_cams:
            percam_ind = torch.where(self.cams == cc)[0]
            uniq_class = torch.unique(self.labels[percam_ind])
            uniq_class = uniq_class[uniq_class >= 0]
            self.concate_intra_class.append(uniq_class)
            cls_mapper = {int(uniq_class[j]): j for j in range(len(uniq_class))}
            self.memory_class_mapper.append(cls_mapper)  # from pseudo label to index under each camera

            percam_label = self.labels[percam_ind]
            percam_feature = all_features[percam_ind]
            cnt = 0
            percam_class_num = len(torch.unique(percam_label[percam_label >= 0]))
            percam_id_feature = torch.zeros((percam_class_num, percam_feature.size(1)))
            for lbl in torch.unique(percam_label):
                if lbl >= 0:
                    ind = torch.where(percam_label == lbl)[0]
                    id_feat = torch.mean(percam_feature[ind], dim=0)
                    percam_id_feature[cnt, :] = id_feat
                    cnt += 1
            percam_id_feature = F.normalize(percam_id_feature, p=2, dim=1)
            self.percam_memory.append(percam_id_feature.cuda().detach())
        self.percam_tempV = torch.cat([feat.detach().clone() for feat in self.percam_memory])
        self.concate_intra_class = torch.cat(self.concate_intra_class)

    def forward(self, inputs, indexes, **kwargs):
        if self.cfg.CAP.NORM_FEAT:
            inputs = F.normalize(inputs, p=2, dim=1)

        indexes = indexes.cuda()
        cams = self.cams[indexes]

        loss_intra = torch.Tensor([0]).cuda()
        loss_inter = torch.Tensor([0]).cuda()
        for cc in torch.unique(cams):
            inds = torch.where(cams == cc)[0]
            percam_targets = self.labels[indexes[inds]]
            percam_feat = inputs[inds]

            # intra-camera loss
            mapped_targets = [self.memory_class_mapper[cc][int(k)] for k in percam_targets]
            mapped_targets = torch.tensor(mapped_targets).cuda()

            percam_inputs = ExemplarMemory.apply(percam_feat, mapped_targets, self.percam_memory[cc], torch.Tensor([self.momentum]).to(percam_feat.device))
            percam_inputs /= self.t  # similarity score before softmax
            loss_intra += F.cross_entropy(percam_inputs, mapped_targets)

            # inter-camera loss
            if self.cur_epoch >= self.intercam_epoch:
                associate_loss = 0
                target_inputs = percam_feat.mm(self.percam_tempV.t().clone())
                temp_sims = target_inputs.detach().clone()
                target_inputs /= self.t

                for k in range(len(percam_feat)):
                    ori_asso_ind = torch.nonzero(self.concate_intra_class == percam_targets[k]).squeeze(-1)
                    temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
                    sel_ind = torch.sort(temp_sims[k])[1][-self.hard_neg_k:]
                    concated_input = torch.cat((target_inputs[k, ori_asso_ind], target_inputs[k, sel_ind]), dim=0)
                    concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).cuda()
                    concated_target[0:len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)
                    associate_loss += -1 * (F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
                loss_inter += associate_loss / len(percam_feat)
        if self.cur_epoch >= self.intercam_epoch:
            return {
                    "loss_intra": loss_intra, 
                    "loss_inter": self.cfg.CAP.LOSS_WEIGHT * loss_inter
                }
        else:
            return {"loss_intra": loss_intra}