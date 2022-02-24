import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, autograd

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
        self.t = self.cfg.CAP.TEMP  # temperature
        self.hard_neg_k = self.cfg.CAP.HARD_NEG_K  # hard negative sampling in global loss
        self.momentum = self.cfg.CAP.MOMENTUM  # momentum update rate
        self.cur_epoch = 0
        self.aploss = APLoss(nq=self.cfg.CAP.LOSS_INSTANCE.NUM_BINS, min=-1, max=1).cuda()
        self.smoothaploss = SmoothAP(self.cfg).cuda()

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
        self.identity_weight = (1 - self.identity_weight / nums).cuda()
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
            sim_target = InstanceMemory.apply(instance_inputs, indexes, self.instance_memory,
                                              torch.Tensor([self.cfg.CAP.LOSS_INSTANCE.MOMENTUM]).to(
                                                  instance_inputs.device))
            value_k_list, gt_k_list = [], []
            for k in range(len(sim_target)):
                sim_k = sim_target[k]
                label_k = self.instance_weight.detach()[indexes][k]
                _, index_k = label_k.sort(descending=True)
                index_k = index_k[:self.cfg.CAP.LOSS_INSTANCE.LIST_LENGTH]
                value_k = sim_k[index_k].unsqueeze(0)
                gt_k = label_k[index_k].unsqueeze(0)
                value_k_list.append(value_k)
                gt_k_list.append(gt_k)
            value_k_list = torch.cat(value_k_list)
            gt_k_list = torch.cat(gt_k_list)
            if self.cfg.CAP.LOSS_INSTANCE.NAME == 'aploss':
                loss_instance += self.aploss(value_k_list, gt_k_list) / len(sim_target)
            elif self.cfg.CAP.LOSS_INSTANCE.NAME == 'smoothaploss':
                loss_instance = self.smoothaploss(value_k_list, gt_k_list) / len(sim_target)
            else:
                raise ValueError

        for cc in torch.unique(cams):
            inds = torch.where(cams == cc)[0]
            percam_targets = self.labels[indexes[inds]]
            percam_feat = inputs[inds]

            # camera loss
            mapped_targets = [self.camera_memory_class_mapper[cc][int(k)] for k in percam_targets]
            mapped_targets = torch.tensor(mapped_targets).cuda()

            percam_inputs = CameraMemory.apply(percam_feat, mapped_targets, self.camera_memory[cc],
                                               torch.Tensor([self.momentum]).to(percam_feat.device))
            percam_inputs /= self.t  # similarity score before softmax
            if self.cfg.CAP.LOSS_CAMERA.WEIGHTED:
                loss_camera += -(F.softmax(self.camera_weight[cc][indexes[inds]].cuda().clone() / self.t,
                                           dim=1) * F.log_softmax(percam_inputs, dim=1)).mean(0).sum()
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
                        concated_target = self.percam_tempW[indexes[inds]][k][
                                              torch.cat([ori_asso_ind, sel_ind.cpu()])] / self.t
                        associate_loss += -1 * (
                                    F.log_softmax(concated_input, dim=0) * F.softmax(concated_target, dim=0)).sum()
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


class APLoss(nn.Module):
    """ Differentiable AP loss, through quantization. From the paper:
        Learning with Average Precision: Training Image Retrieval with a Listwise Loss
        Jerome Revaud, Jon Almazan, Rafael Sampaio de Rezende, Cesar de Souza
        https://arxiv.org/abs/1906.07589
        Input: (N, M)   values in [min, max]
        label: (N, M)   values in {0, 1}
        Returns: 1 - mAP (mean AP for each n in {1..N})
                 Note: typically, this is what you wanna minimize
    """

    def __init__(self, nq=25, min=0, max=1):
        nn.Module.__init__(self)
        assert isinstance(nq, int) and 2 <= nq <= 100
        self.nq = nq
        self.min = min
        self.max = max
        gap = max - min
        assert gap > 0
        # Initialize quantizer as non-trainable convolution
        self.quantizer = q = nn.Conv1d(1, 2 * nq, kernel_size=1, bias=True)
        q.weight = nn.Parameter(q.weight.detach(), requires_grad=False)
        q.bias = nn.Parameter(q.bias.detach(), requires_grad=False)
        a = (nq - 1) / gap
        # First half equal to lines passing to (min+x,1) and (min+x+1/a,0) with x = {nq-1..0}*gap/(nq-1)
        q.weight[:nq] = -a
        q.bias[:nq] = torch.from_numpy(a * min + np.arange(nq, 0, -1))  # b = 1 + a*(min+x)
        # First half equal to lines passing to (min+x,1) and (min+x-1/a,0) with x = {nq-1..0}*gap/(nq-1)
        q.weight[nq:] = a
        q.bias[nq:] = torch.from_numpy(np.arange(2 - nq, 2, 1) - a * min)  # b = 1 - a*(min+x)
        # First and last one as a horizontal straight line
        q.weight[0] = q.weight[-1] = 0
        q.bias[0] = q.bias[-1] = 1

    def forward(self, x, label, qw=None, ret='1-mAP'):
        assert x.shape == label.shape  # N x M
        N, M = x.shape
        # Quantize all predictions
        q = self.quantizer(x.unsqueeze(1))
        q = torch.min(q[:, :self.nq], q[:, self.nq:]).clamp(min=0)  # N x Q x M

        nbs = q.sum(dim=-1)  # number of samples  N x Q = c
        rec = (q * label.view(N, 1, M).float()).sum(dim=-1)  # number of correct samples = c+ N x Q
        prec = rec.cumsum(dim=-1) / (1e-16 + nbs.cumsum(dim=-1))  # precision
        rec /= rec.sum(dim=-1).unsqueeze(1)  # norm in [0,1]

        ap = (prec * rec).sum(dim=-1)  # per-image AP

        if ret == '1-mAP':
            if qw is not None:
                ap *= qw  # query weights
            return 1 - ap.mean()
        elif ret == 'AP':
            assert qw is None
            return ap
        else:
            raise ValueError("Bad return type for APLoss(): %s" % str(ret))


class SmoothAP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.anneal = 0.01
        self.thresh = cfg.CAP.LOSS_INSTANCE.THRESH
        self.cfg = cfg

    def forward(self, sim_dist, targets):
        # ------ differentiable ranking of all retrieval set ------
        N, M = sim_dist.size()
        # Compute the mask which ignores the relevance score of the query to itself
        mask_indx = 1.0 - torch.eye(M, device=sim_dist.device)
        mask_indx = mask_indx.unsqueeze(dim=0).repeat(N, 1, 1)  # (N, M, M)

        sim_dist_repeat = sim_dist.unsqueeze(dim=1).repeat(1, M, 1)  # (N, M, M)
        sim_diff = sim_dist_repeat - sim_dist_repeat.permute(0, 2, 1)  # (N, M, M)
        # Pass through the sigmoid
        sim_sg = sigmoid(sim_diff, temp=self.anneal) * mask_indx
        # sim_gt = (sim_diff > 0).float() * mask_indx
        # gt_rank = sim_gt.sum(dim=-1) + 1
        # Compute all the rankings
        sim_all_rk = torch.sum(sim_sg, dim=-1) + 1  # (N, M)

        pos_mask = (targets > self.thresh).float()  # * targets
        pos_mask_repeat = pos_mask.unsqueeze(1).repeat(1, M, 1)

        # Compute positive rankings
        pos_sim_sg = sim_sg * pos_mask_repeat
        sim_pos_rk = torch.sum(pos_sim_sg, dim=-1) + 1  # (N, M)

        # sum the values of the Smooth-AP for all instances in the mini-batch
        ap = 0
        if self.cfg.CAP.LOSS_INSTANCE.SMOOTHAP_TARGET:
            pos_divide_all = sim_pos_rk * pos_mask * targets / sim_all_rk
        else:
            pos_divide_all = sim_pos_rk * pos_mask / sim_all_rk

        for ind in range(N):
            pos_divide = torch.sum(pos_divide_all[ind])
            if self.cfg.CAP.LOSS_INSTANCE.SMOOTHAP_TARGET:
                ap += pos_divide / torch.sum(pos_mask[ind] * targets[ind]) / N
            else:
                ap += pos_divide / torch.sum(pos_mask[ind]) / N
        return 1 - ap


def sigmoid(tensor, temp=1.0):
    """ temperature controlled sigmoid
    takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    # clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y
