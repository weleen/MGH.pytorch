import numpy as np
import torch
from torch import nn, autograd
import torch.nn.functional as F
from torch.autograd import Variable

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
        self.aploss = APLoss(nq=self.cfg.CAP.LOSS_INSTANCE.NUM_BINS, min=-1, max=1).cuda()
        self.fastaploss = FastAPLoss(num_bins=self.cfg.CAP.LOSS_INSTANCE.NUM_BINS).cuda()

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
            sim_target = InstanceMemory.apply(instance_inputs, indexes, self.instance_memory, torch.Tensor([self.momentum]).to(instance_inputs.device))
            # sim_target /= self.cfg.CAP.LOSS_INSTANCE.TEMP
            # gt_perm = torch.argsort(self.instance_weight.detach()[indexes], descending=True, dim=1)
            if self.cfg.CAP.LOSS_INSTANCE.NAME == 'aploss':
                for k in range(len(sim_target)):
                    sim_k = sim_target[k]
                    label_k = self.instance_weight.detach()[indexes][k]
                    _, index_k = label_k.sort(descending=True)
                    index_k = index_k[:self.cfg.CAP.LOSS_INSTANCE.LIST_LENGTH]
                    value_k = sim_k[index_k].unsqueeze(0)
                    gt_k = label_k[index_k].unsqueeze(0)
                    loss_instance += self.aploss(value_k, gt_k) / len(sim_target)
                    # loss_instance += self.fastaploss(value_k, label_k)
                    # loss_instance += self.fastaploss(value_k, self.instance_labels[index_k])
            elif self.cfg.CAP.LOSS_INSTANCE.NAME == 'fastaploss':
                # loss_instance = self.fastaploss(instance_inputs, self.instance_labels[indexes])
                loss_instance = self.fastaploss(sim_target, self.instance_weight.detach()[indexes])
                # loss_instance = self.aploss(sim_target, self.instance_weight.detach()[indexes])
            else:
                raise ValueError

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
                        concated_target = self.percam_tempW[indexes[inds]][k][torch.cat([ori_asso_ind, sel_ind.cpu()])] / self.t
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
        self.quantizer = q = nn.Conv1d(1, 2*nq, kernel_size=1, bias=True)
        q.weight = nn.Parameter(q.weight.detach(), requires_grad=False)
        q.bias = nn.Parameter(q.bias.detach(), requires_grad=False)
        a = (nq-1) / gap
        # First half equal to lines passing to (min+x,1) and (min+x+1/a,0) with x = {nq-1..0}*gap/(nq-1)
        q.weight[:nq] = -a
        q.bias[:nq] = torch.from_numpy(a*min + np.arange(nq, 0, -1))  # b = 1 + a*(min+x)
        # First half equal to lines passing to (min+x,1) and (min+x-1/a,0) with x = {nq-1..0}*gap/(nq-1)
        q.weight[nq:] = a
        q.bias[nq:] = torch.from_numpy(np.arange(2-nq, 2, 1) - a*min)  # b = 1 - a*(min+x)
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


def softBinning(D, mid, Delta):
    y = 1 - torch.abs(D-mid)/Delta
    return torch.max(torch.Tensor([0]).cuda(), y)

def dSoftBinning(D, mid, Delta):
    side1 = (D > (mid - Delta)).type(torch.float)
    side2 = (D <= mid).type(torch.float)
    ind1 = (side1 * side2) #.type(torch.uint8)

    side1 = (D > mid).type(torch.float)
    side2 = (D <= (mid + Delta)).type(torch.float)
    ind2 = (side1 * side2) #.type(torch.uint8)

    return (ind1 - ind2)/Delta
    

class FastAPLoss(torch.nn.Module):
    """
    FastAP - loss layer definition
    This class implements the FastAP loss from the following paper:
    "Deep Metric Learning to Rank", 
    F. Cakir, K. He, X. Xia, B. Kulis, S. Sclaroff. CVPR 2019
    """
    def __init__(self, num_bins=10):
        super(FastAPLoss, self).__init__()
        self.num_bins = num_bins

    def forward(self, batch, labels):
        return FastAP.apply(batch, labels, self.num_bins)


class FastAP(torch.autograd.Function):
    """
    FastAP - autograd function definition
    This class implements the FastAP loss from the following paper:
    "Deep Metric Learning to Rank", 
    F. Cakir, K. He, X. Xia, B. Kulis, S. Sclaroff. CVPR 2019
    NOTE:
        Given a input batch, FastAP does not sample triplets from it as it's not 
        a triplet-based method. Therefore, FastAP does not take a Sampler as input. 
        Rather, we specify how the input batch is selected.
    """

    @staticmethod
    def forward(ctx, input, target, num_bins):
        """
        Args:
            input:     torch.Tensor(N x M), similarity matrix
            target:    torch.Tensor(N x M), relevance matrix
            num_bins:  int, number of bins in distance histogram
        """
        N = target.size()[0]
        neg_target = 1 - target
        assert input.size()[0] == N, "Batch size donesn't match!"
        
        # 1. get affinity matrix
        I_pos = target
        I_neg = neg_target
        N_pos = torch.sum(target, 1)
        
        # 2. compute distances from embeddings
        # squared Euclidean distance with range [0,4]
        dist2 = 2 - 2 * input.clamp(-1, 1)

        # 3. estimate discrete histograms
        Delta = torch.tensor(4. / num_bins).cuda()
        Z     = torch.linspace(0., 4., steps=num_bins+1).cuda()
        L     = Z.size()[0]
        h_pos = torch.zeros((N, L)).cuda()
        h_neg = torch.zeros((N, L)).cuda()
        for l in range(L):
            pulse    = softBinning(dist2, Z[l], Delta)
            h_pos[:,l] = torch.sum(pulse * I_pos, 1)
            h_neg[:,l] = torch.sum(pulse * I_neg, 1)

        H_pos = torch.cumsum(h_pos, 1)
        h     = h_pos + h_neg
        H     = torch.cumsum(h, 1)
        
        # 4. compate FastAP
        FastAP = h_pos * H_pos / H
        FastAP[torch.isnan(FastAP) | torch.isinf(FastAP)] = 0
        FastAP = torch.sum(FastAP,1)/N_pos
        FastAP = FastAP[ ~torch.isnan(FastAP) ]
        loss   = 1 - torch.mean(FastAP)
        # if torch.rand(1) > 0.99:
        print("loss value (1-mean(FastAP)): ", loss.item())

        # 6. save for backward
        ctx.save_for_backward(input, target)
        ctx.Z     = Z
        ctx.Delta = Delta
        ctx.dist2 = dist2
        ctx.I_pos = I_pos
        ctx.I_neg = I_neg
        ctx.h_pos = h_pos
        ctx.h_neg = h_neg
        ctx.H_pos = H_pos
        ctx.N_pos = N_pos
        ctx.h     = h
        ctx.H     = H
        ctx.L     = torch.tensor(L)
        
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_tensors

        Z     = Variable(ctx.Z     , requires_grad = False)
        Delta = Variable(ctx.Delta , requires_grad = False)
        dist2 = Variable(ctx.dist2 , requires_grad = False)
        I_pos = Variable(ctx.I_pos , requires_grad = False)
        I_neg = Variable(ctx.I_neg , requires_grad = False)
        h     = Variable(ctx.h     , requires_grad = False)
        H     = Variable(ctx.H     , requires_grad = False)
        h_pos = Variable(ctx.h_pos , requires_grad = False)
        h_neg = Variable(ctx.h_neg , requires_grad = False)
        H_pos = Variable(ctx.H_pos , requires_grad = False)
        N_pos = Variable(ctx.N_pos , requires_grad = False)
        print('backward test')
        L     = Z.size()[0]
        H2    = torch.pow(H,2)
        H_neg = H - H_pos

        # 1. d(FastAP)/d(h+)
        LTM1 = torch.tril(torch.ones(L,L), -1)  # lower traingular matrix
        tmp1 = h_pos * H_neg / H2
        tmp1[torch.isnan(tmp1)] = 0

        d_AP_h_pos = (H_pos * H + h_pos * H_neg) / H2 
        d_AP_h_pos = d_AP_h_pos + torch.mm(tmp1, LTM1.cuda())
        d_AP_h_pos = d_AP_h_pos / N_pos.repeat(L,1).t()
        d_AP_h_pos[torch.isnan(d_AP_h_pos) | torch.isinf(d_AP_h_pos)] = 0


        # 2. d(FastAP)/d(h-)
        LTM0 = torch.tril(torch.ones(L,L), 0)  # lower triangular matrix
        tmp2 = -h_pos * H_pos / H2
        tmp2[torch.isnan(tmp2)] = 0

        d_AP_h_neg = torch.mm(tmp2, LTM0.cuda())
        d_AP_h_neg = d_AP_h_neg / N_pos.repeat(L,1).t()
        d_AP_h_neg[torch.isnan(d_AP_h_neg) | torch.isinf(d_AP_h_neg)] = 0


        # 3. d(FastAP)/d(embedding)
        d_AP_x = 0
        for l in range(L):
            dpulse = dSoftBinning(dist2, Z[l], Delta)
            dpulse[torch.isnan(dpulse) | torch.isinf(dpulse)] = 0
            ddp = dpulse * I_pos
            ddn = dpulse * I_neg
            alpha_p = torch.diag(d_AP_h_pos[:,l]) # N*N
            alpha_n = torch.diag(d_AP_h_neg[:,l])
            Ap = torch.mm(ddp, alpha_p) + torch.mm(alpha_p, ddp)
            An = torch.mm(ddn, alpha_n) + torch.mm(alpha_n, ddn)
            
            # accumulate gradient 
            d_AP_x = d_AP_x - torch.mm(input.t(), (Ap+An))
            
        grad_input = -d_AP_x
        return grad_input.t(), None, None