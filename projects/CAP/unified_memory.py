import torch
from torch import nn, autograd
import torch.nn.functional as F

from fastreid.utils.comm import all_gather_tensor, all_gather, get_world_size


class InstanceMemory(autograd.Function):
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


class CameraMemory(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        outputs = inputs.mm(ctx.features.t())
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


class IdentityMemory(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        outputs = inputs.mm(ctx.features.t())
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


class UnifiedMemory(nn.Module):
    def __init__(self, cfg):
        super(UnifiedMemory, self).__init__()
        self.cfg = cfg
        self.t = self.cfg.CAP.TEMP # temperature
        self.hard_neg_k = self.cfg.CAP.HARD_NEG_K # hard negative sampling in global loss
        self.momentum = self.cfg.CAP.MOMENTUM # momentum update rate
        self.cur_epoch = 0

    def _update_epoch(self, epoch):
        self.cur_epoch = epoch

    @torch.no_grad()
    def _update_centers_and_labels(self, all_features: list, all_labels: list, all_camids: list, all_dist_mat: list):
        assert len(all_features) == len(all_labels) == 1, "Support only one dataset."
        features = all_features[0]
        dist_mat = all_dist_mat[0]
        self.cams = torch.cat(all_camids)
        self.labels = torch.tensor(all_labels[0])

        # instance memory and weight
        self.instance_memory = F.normalize(features, p=2, dim=1)
        self.instance_labels = self.labels.clone()
        self.instance_weight = (1 - dist_mat)

        # camera aware memory and weight
        self.camera_memory = []
        self.camera_weight = []
        self.camera_memory_class_mapper = []
        self.concate_intra_class = []
        for cc in torch.unique(self.cams):
            percam_ind = torch.where(self.cams == cc)[0]
            uniq_class = torch.unique(self.labels[percam_ind])
            uniq_class = uniq_class[uniq_class >= 0]
            self.concate_intra_class.append(uniq_class)
            cls_mapper = {int(uniq_class[j]): j for j in range(len(uniq_class))}
            self.camera_memory_class_mapper.append(cls_mapper)  # from pseudo label to index under each camera

            percam_label = self.labels[percam_ind]
            percam_feature = features[percam_ind]
            cnt = 0
            percam_class_num = len(uniq_class)
            percam_id_feature = torch.zeros((percam_class_num, percam_feature.size(1)))
            for lbl in torch.unique(percam_label):
                if lbl >= 0:
                    ind = torch.where(percam_label == lbl)[0]
                    id_feat = torch.mean(percam_feature[ind], dim=0)
                    percam_id_feature[cnt, :] = id_feat
                    cnt += 1
            percam_id_feature = F.normalize(percam_id_feature, p=2, dim=1)
            self.camera_memory.append(percam_id_feature.detach().cuda())
            # camera aware weight
            camera_weight = torch.zeros((len(features), percam_class_num))
            percam_nums = torch.zeros((1, percam_class_num))
            percam_pos_ind = torch.where(percam_label >= 0)[0]
            percam_mapped_label = torch.tensor([cls_mapper[int(i)] for i in percam_label[percam_pos_ind]])
            percam_dist_mat = dist_mat[percam_ind[percam_pos_ind]].t()
            camera_weight.index_add_(1, percam_mapped_label, percam_dist_mat)
            percam_nums.index_add_(1, percam_mapped_label, torch.ones(1, len(percam_pos_ind)))
            camera_weight = 1 - camera_weight / percam_nums
            self.camera_weight.append(camera_weight.cuda())

        self.percam_tempV = torch.cat([feat.detach().clone() for feat in self.camera_memory]).cuda()
        self.concate_intra_class = torch.cat(self.concate_intra_class)
        self.percam_tempW = torch.cat([feat.detach().clone() for feat in self.camera_weight], dim=1).cuda()

        # identity memory and weight
        num_classes = len(torch.unique(self.labels[self.labels >= 0]))
        self.identity_memory = torch.zeros((num_classes, self.instance_memory.size(1)))
        self.identity_weight = torch.zeros((len(self.labels), num_classes))
        nums = torch.zeros((1, num_classes))
        index_select = torch.where(self.labels >= 0)[0]
        inputs_select = dist_mat[index_select].t()
        features_select = self.instance_memory[index_select]
        labels_select = self.labels[index_select]
        self.identity_memory.index_add_(0, labels_select, features_select)
        nums.index_add_(1, labels_select, torch.ones(1, len(index_select)))

        self.identity_weight.index_add_(1, labels_select, inputs_select)
        self.identity_weight  = (1 - self.identity_weight / nums).cuda()
        self.identity_memory /= nums.t()
        self.identity_memory = F.normalize(self.identity_memory, p=2, dim=1).cuda()

        # convert to gpu device
        self.instance_memory = self.instance_memory.cuda()
        self.instance_weight = self.instance_weight.cuda()
        self.instance_labels = self.instance_labels.cuda()

    def forward(self, inputs, indexes, **kwargs):
        if self.cfg.CAP.NORM_FEAT:
            inputs = F.normalize(inputs, p=2, dim=1)
        instance_inputs = inputs.clone()
        indexes = indexes.cuda()
        cams = self.cams[indexes]

        loss_identity = torch.Tensor([0]).cuda()
        loss_camera = torch.Tensor([0]).cuda()
        loss_instance = torch.Tensor([0]).cuda()

        # instance loss
        if self.cur_epoch >= self.cfg.CAP.LOSS_INSTANCE.START_EPOCH:
            targets = self.instance_labels[indexes.cpu()].cuda()
            sim_target = InstanceMemory.apply(instance_inputs, targets, self.instance_memory, torch.Tensor([self.momentum]).to(instance_inputs.device))
            sim_target /= self.cfg.CAP.LOSS_INSTANCE.TEMP
            gt_perm = torch.argsort(self.instance_weight.detach()[indexes], descending=True, dim=1)
            for i in range(len(instance_inputs)):
                perm_prob = _get_perm_prob(sim_target[i], gt_perm[i][1:self.cfg.CAP.LOSS_INSTANCE.LIST_LENGTH+1].cpu().tolist())
                loss_instance += -torch.log(perm_prob)
            loss_instance /= len(instance_inputs)

        for cc in torch.unique(cams):
            inds = torch.where(cams == cc)[0]
            percam_targets = self.labels[indexes[inds]]
            percam_feat = inputs[inds]

            # camera loss
            mapped_targets = [self.camera_memory_class_mapper[cc][int(k)] for k in percam_targets]
            mapped_targets = torch.tensor(mapped_targets).cuda()

            percam_inputs = CameraMemory.apply(percam_feat, mapped_targets, self.camera_memory[cc], torch.Tensor([self.momentum]).to(percam_feat.device))
            percam_inputs /= self.t  # similarity score before softmax
            if self.cfg.CAP.LOSS_CAMERA.WEIGHTED:
                loss_camera += -(F.softmax(self.camera_weight[cc][indexes[inds]].cuda().clone() / self.t, dim=1) * F.log_softmax(percam_inputs, dim=1)).mean(0).sum()
            else:
                loss_camera += F.cross_entropy(percam_inputs, mapped_targets)

            # identity loss
            if self.cur_epoch >= self.cfg.CAP.LOSS_IDENTITY.START_EPOCH:
                associate_loss = 0
                target_inputs = percam_feat.mm(self.percam_tempV.t().clone())
                temp_sims = target_inputs.detach().clone()
                target_inputs /= self.t

                for k in range(len(percam_feat)):
                    ori_asso_ind = torch.where(self.concate_intra_class == percam_targets[k])[0]
                    temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
                    if self.cfg.CAP.ENABLE_HARD_NEG:
                        sel_ind = torch.sort(temp_sims[k])[1][-self.hard_neg_k:]
                    else:
                        sel_ind = torch.sort(temp_sims[k])[1]
                        sel_ind = sel_ind[-(len(sel_ind) - len(ori_asso_ind)):]
                    concated_input = torch.cat((target_inputs[k, ori_asso_ind], target_inputs[k, sel_ind]), dim=0)
                    if self.cfg.CAP.LOSS_IDENTITY.WEIGHTED:
                        concated_target = self.percam_tempW[indexes[inds]][k][torch.cat([ori_asso_ind, sel_ind])].cuda() / self.t
                        associate_loss += -1 * (F.log_softmax(concated_input, dim=0) * F.softmax(concated_target, dim=0)).sum()
                    else:
                        concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).cuda()
                        concated_target[0:len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)
                        associate_loss += -1 * (F.log_softmax(concated_input, dim=0) * concated_target).sum()
                loss_identity += associate_loss / len(percam_feat)

        loss_dict = dict()
        if self.cur_epoch >= self.cfg.CAP.LOSS_INSTANCE.START_EPOCH:
            loss_dict.update({
                "loss_instance": loss_instance * self.cfg.CAP.LOSS_INSTANCE.SCALE
            })
        if self.cur_epoch >= self.cfg.CAP.LOSS_IDENTITY.START_EPOCH:
            loss_dict.update({
                "loss_identity": loss_identity * self.cfg.CAP.LOSS_IDENTITY.SCALE
            })
        if self.cur_epoch >= self.cfg.CAP.LOSS_CAMERA.START_EPOCH:
            loss_dict.update({
                "loss_camera": loss_camera * self.cfg.CAP.LOSS_CAMERA.SCALE
            })
        return loss_dict

from itertools import permutations

def _get_perm_prob(scores, perm, eps=1e-6):
    exp_score_sum = 0
    perm_prob = 1
    for idx in reversed(perm):
        exp_score = torch.exp(scores[idx])
        exp_score_sum += exp_score
        perm_prob *= exp_score / (exp_score_sum + eps)
    return perm_prob

def _get_all_perm_prob(scores, indices):
    return [_get_perm_prob(scores, r) for r in sorted(permutations(indices))]