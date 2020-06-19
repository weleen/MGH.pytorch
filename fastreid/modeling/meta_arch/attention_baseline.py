import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.utils import make_grid

from fastreid.modeling.meta_arch.build import META_ARCH_REGISTRY
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_reid_heads
from fastreid.modeling.losses import reid_losses, WeakInlierCountPool, TransformedGridLoss
from fastreid.layers import GeneralizedMeanPoolingP
from fastreid.utils.torch_tool import denorm, visualize_cam
from fastreid.utils.geometric_transform import GeometricTnf


@META_ARCH_REGISTRY.register()
class CAMBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        self.device = cfg.MODEL.DEVICE
        # backbone
        self.backbone = build_backbone(cfg)

        # head
        if cfg.MODEL.HEADS.POOL_LAYER == 'avgpool':
            pool_layer = nn.AdaptiveAvgPool2d(1)
        elif cfg.MODEL.HEADS.POOL_LAYER == 'maxpool':
            pool_layer = nn.AdaptiveMaxPool2d(1)
        elif cfg.MODEL.HEADS.POOL_LAYER == 'gempool':
            pool_layer = GeneralizedMeanPoolingP()
        else:
            pool_layer = nn.Identity()
        self.in_feat = cfg.MODEL.HEADS.IN_FEAT
        self.heads = build_reid_heads(cfg, self.in_feat, pool_layer)

        # cam related features
        self.forward_features = None
        self.backward_features = None
        self._register_module_hook(grad_layer='layer4.2')
        self._image_buffer = None

        # transformation related
        self.temperature = 512 ** -0.5
        self.num_instance = cfg.DATALOADER.NUM_INSTANCE
        self.afterconv1 = nn.Conv2d(self.in_feat, 512, kernel_size=1, bias=False)

        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)
        self.relu = nn.ReLU(inplace=True)

        self.feat_size = (cfg.INPUT.SIZE_TRAIN[0] // 16, cfg.INPUT.SIZE_TRAIN[1] // 16)
        self.trans_net = nn.Sequential(
            nn.Conv2d(self.feat_size[0] * self.feat_size[1], 64, kernel_size=4, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 32, kernel_size=4, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )  # predict transformation theta parameters
        corrdim_trans = 32 * 10 * 2
        if self._cfg.INPUT.GEOMETRIC == 'affine':
            geometric_param = 6
        else:
            geometric_param = 18
        self.linear = nn.Linear(corrdim_trans, geometric_param)
        self.geometricTnf = GeometricTnf(geometric_model=self._cfg.INPUT.GEOMETRIC,
                                         out_h=self.feat_size[0],
                                         out_w=self.feat_size[1],
                                         offset_factor=227 / 210)
        self.criterion_inlier = WeakInlierCountPool(geometric_model=self._cfg.INPUT.GEOMETRIC,
                                                    tps_grid_size=3, tps_reg_factor=0.2,
                                                    h_matches=self.feat_size[0], w_matches=self.feat_size[1],
                                                    use_conv_filter=False, dilation_filter=0,
                                                    normalize_inlier_count=True)
        self.criterion_synth = TransformedGridLoss(geometric_model=self._cfg.INPUT.GEOMETRIC, use_cuda=True)

    def _register_module_hook(self, grad_layer):
        def forward_hook(module, input, output):
            self.forward_features = output

        def backward_hook(module, grad_input, grad_output):
            self.backward_features = grad_output[0]

        gradient_layer_found = False
        for idx, m in self.backbone.named_modules():
            if idx == grad_layer:
                print('Register hook for {}'.format(grad_layer))
                m.register_forward_hook(forward_hook)
                m.register_backward_hook(backward_hook)
                gradient_layer_found = True
                break

        # for our own sanity, confirm its existence
        if not gradient_layer_found:
            raise AttributeError('Gradient layer %s not found in the internal model' % grad_layer)

    def forward(self, inputs):
        images, targets = self.preprocess(inputs)

        if not self.training:
            # inference
            bn_feat = self.inference(images)
            return bn_feat, targets, inputs["camid"]

        features = self.backbone(images)  # (bs, 2048, 16, 8)

        # get the guided attentive mask
        mask = self.get_cam(images, features, targets)

        # forward the features into head, this head is in train mode
        logits, global_feat, bn_feat = self.heads(features, targets)

        # get the masked images
        masked_images = images.clone() * (1 - mask.to(images.device))
        # # forward the masked image again, take care of the memory cost
        # features_am = self.backbone(masked_images)
        # logits_am, global_feat_am, bn_feat_am = self.heads(features_am)

        # perform affine transformation estimation, synthetic images, and corresponding theta,
        # select the first instance as the target
        images_affine = inputs['images_affine']
        theta_gt = inputs['theta']
        images_affine = images_affine.view(-1, self.num_instance, *images_affine.size()[1:])[:, 0]
        theta_gt = theta_gt.view(-1, self.num_instance, *theta_gt.size()[1:])[:, 0]

        # group the instances
        features_inst = features.view(-1, self.num_instance, *features.size()[1:])

        target_feature = features_inst[:, 0].contiguous()
        target_feature_affine = self.backbone(images_affine)
        src_feature = features_inst[:, 1:].contiguous()
        # normalize features
        target_feature_norm = F.normalize(target_feature, p=2, dim=1)
        target_feature_affine_norm = F.normalize(target_feature_affine, p=2, dim=1)
        src_feature_norm = F.normalize(src_feature, p=2, dim=2)

        b, t, c, h, w = src_feature.size()
        # calculate transformation from src to target_affine
        corrfeat1, corrfeat1_trans, trans1_out = self.compute_tnf_src_to_target_affine(
            target_feature_affine_norm, src_feature_norm)
        src_feature_ori = src_feature.view(b * t, c, h, w)
        src_feature_transformed = self.geometricTnf(src_feature_ori, trans1_out)  # b x t, c, h, w

        # calculate transformation from affine to transformed_src
        src_feature_transformed_norm = F.normalize(src_feature_transformed, p=2, dim=1).view(b, t, c, h, w)
        corrfeat2, corrfeat2_trans, trans2_out = self.compute_tnf_target_to_src_transfromed(
            target_feature_norm, src_feature_transformed_norm)

        # recurrent align
        def recurrent_align(init_query, idx):
            trans_thetas = []
            trans_feats = []
            cur_query = init_query
            for t in idx:
                cur_base_norm = src_feature_norm[:, t: t + 1]  # b, 1, c, h, w
                cur_base_feat = src_feature[:, t]  # b, c, h, w
                corrfeat, corrfeat_trans, trans_theta = \
                    self.compute_tnf_src_to_target_affine(cur_query, cur_base_norm)
                cur_base_transformed = self.geometricTnf(cur_base_feat, trans_theta)
                cur_query = F.normalize(cur_base_transformed, p=2, dim=1)
                trans_thetas.append(trans_theta)
                trans_feats.append(cur_base_transformed)

            return trans_thetas, trans_feats

        # cycle transformation
        def cycle(TT):
            # propagate backward
            backward_trans_thetas, backward_trans_feats = \
                recurrent_align(target_feature_affine_norm, list(range(t))[::-1][:TT])
            backward_trans_feats_norm = F.normalize(backward_trans_feats[-1], p=2, dim=1)
            forward_trans_thetas, forward_trans_feats = \
                recurrent_align(backward_trans_feats_norm, list(range(t))[t - TT + 1:])
            # cycle back from last src frame to target
            last_ = forward_trans_feats[-1] if len(forward_trans_feats) > 0 \
                else backward_trans_feats[0]
            last_corrfeat, last_corrfeat_trans, last_trans_theta = \
                self.compute_tnf_src_to_target_affine(
                    F.normalize(last_, p=2, dim=1), target_feature_norm.unsqueeze(1))
            forward_trans_thetas.append(last_trans_theta)

            return backward_trans_thetas, forward_trans_thetas, backward_trans_feats

        cycle_outputs = [[], [], []]
        for c in range(1, t + 1):
            _output = cycle(c)
            for i, o in enumerate(_output):
                cycle_outputs[i] += o
        #     if c == t:
        #         back_trans_feats = _output[-1]
        #
        # back_trans_feats = torch.stack(back_trans_feats).transpose(0, 1).contiguous()  # b, t, c, h, w
        # back_trans_feats = F.normalize(back_trans_feats, p=2, dim=2)
        # skip_corrfeat, skip_trans, skip_corrfeat_mat = \
        #     self.compute_tnf_target_to_src_transfromed(target_feature_norm, back_trans_feats)

        # get the guided attentive mask for target_images
        # visualization
        mask_affine = self.get_cam(images_affine, target_feature_affine, targets.view(-1, self.num_instance)[:, 0])
        masked_images_affine = images_affine.clone() * (1 - mask_affine.to(images_affine.device))
        # transform the masks
        mask_transformed = self.geometricTnf(mask.view(b, self.num_instance, *mask.size()[1:])[:, 0],
                                             trans2_out.view(b, t, -1)[:, 0],
                                             out_h=mask.size(2),
                                             out_w=mask.size(3))
        masked_images_transformed = images_affine.clone() * (1 - mask_transformed.to(images_affine.device))
        # save the images for visualization
        mask_vis = mask.view(-1, self.num_instance, *mask.size()[1:])[:, 0]
        images_vis = images.view(-1, self.num_instance, *images.size()[1:])[:, 0]
        masked_images_vis = masked_images.view(-1, self.num_instance, *masked_images.size()[1:])[:, 0]
        self.save_context({'images': images_vis.cpu(),
                           'masked_images': masked_images_vis.cpu(),
                           'mask': mask_vis.cpu(),
                           'images_affine': images_affine.cpu(),
                           'masked_images_affine': masked_images_affine.cpu(),
                           'mask_affine': mask_affine.cpu(),
                           'masked_images_transformed': masked_images_transformed.cpu(),
                           'mask_transformed': mask_transformed.cpu(),
                           'image_path': np.asarray(inputs['img_path'])[
                               list((range(0, len(inputs['img_path']), self.num_instance)))].tolist()})
        return logits, global_feat, targets, \
               cycle_outputs[:2], target_feature_affine, theta_gt, trans1_out, trans2_out, corrfeat1
               # mask_transformed, src_feature_transformed,
               # logits_am, global_feat_am

    def save_context(self, context: dict):
        self._image_buffer = context

    def inference(self, images):
        assert not self.training
        features = self.backbone(images)  # (bs, 2048, 16, 8)
        bn_feat = self.heads(features)
        return bn_feat

    def losses(self, outputs):
        loss_dict = {}
        logits, global_feat, targets, \
        cycle_outputs, target_feature_affine, theta_affine, trans1_out, trans2_out, corrfeat1 = \
            outputs
        loss_dict.update(reid_losses(self._cfg, logits, global_feat, targets))
        # correspondence related loss
        back_trans_thetas, forw_trans_thetas = cycle_outputs
        nn = list(range(len(forw_trans_thetas)))
        nn = [ii for ii in [sum(nn[:i]) - 1 for i in nn][2:] if ii < len(forw_trans_thetas)]
        # forward theta loss
        loss_forward_theta = []
        loss_targ_theta_skip = []
        for i in nn:
            loss_forward_theta.append(self.criterion_synth(forw_trans_thetas[i], theta_affine))

        theta2 = theta_affine.unsqueeze(1).repeat(1, trans2_out.size(0) // theta_affine.size(0), 1, 1).view(-1, 2, 3)

        loss_targ_theta_skip.append(self.criterion_synth(trans2_out, theta2))

        loss_inlier = self.criterion_inlier(matches=corrfeat1, theta=trans1_out)
        loss_inlier = torch.mean(-loss_inlier)
        # attentive mining loss
        # loss_dict.update(
        #     {"loss_am": (F.softmax(logits_am, dim=1) * F.one_hot(targets, logits.size(1))).mean(0).sum()})
        # geometric loss
        loss_dict.update({
            "loss_foward_theta": sum(loss_forward_theta),
            "loss_targ_theta_skip": sum(loss_targ_theta_skip),
            "loss_inlier": loss_inlier
        })
        return loss_dict

    def preprocess(self, inputs):
        images = inputs['images'].to(self.device)
        targets = inputs['targets'].to(self.device)
        return images, targets

    def get_image_buffer(self):
        images = self._image_buffer['images']
        masked_images = self._image_buffer['masked_images']
        mask = self._image_buffer['mask'].detach()
        images_affine = self._image_buffer['images_affine']
        masked_images_affine = self._image_buffer['masked_images_affine']
        mask_affine = self._image_buffer['mask_affine']
        masked_images_transformed = self._image_buffer['masked_images_transformed']
        mask_transformed = self._image_buffer['mask_transformed']
        image_path = self._image_buffer['image_path']

        buffer = dict()
        # denormalize the images
        denorm_images = denorm(images, self._cfg.MODEL.PIXEL_MEAN, self._cfg.MODEL.PIXEL_STD)
        denorm_masked_images = denorm(masked_images, self._cfg.MODEL.PIXEL_MEAN, self._cfg.MODEL.PIXEL_STD)
        denorm_images_affine = denorm(images_affine, self._cfg.MODEL.PIXEL_MEAN, self._cfg.MODEL.PIXEL_STD)

        heatmap_images = []
        blend_images = []
        for i in range(mask.size(0)):
            m = mask[i].unsqueeze(0)
            heatmap_sm, result_sm = visualize_cam(m, denorm_images[i].unsqueeze(0))
            heatmap_images.append(heatmap_sm)
            blend_images.append(result_sm)
        heatmap_images = torch.stack(heatmap_images, 0)
        blend_images = torch.stack(blend_images, 0)

        for h, im, mask_im, blend_im, im_path in zip(heatmap_images, denorm_images, denorm_masked_images,
                                                     blend_images, image_path):
            im_path = im_path.split('/')[-1].strip('.jpg').strip('.png')
            grid_im = make_grid(torch.stack([im, h, blend_im, mask_im]), nrow=4)
            buffer.update({'images/{}'.format(im_path): grid_im})
        self._image_buffer.clear()
        return buffer

    def get_cam(self, images, features, targets, sigma=0.25, w=20):
        # set bn in heads to eval mode, or the mask is wrong,
        # after obtaining the mask, we will forward the feature again.
        self.heads.eval()
        assert not self.heads.training
        # forward the original image
        # this is ugly, due to the train/eval is different for self.heads
        bn_feat = self.heads(features, targets)
        try:
            logits = self.heads.classifier(bn_feat)
        except:
            logits = self.heads.classifier(bn_feat, targets)

        labels_one = F.one_hot(targets, num_classes=self._cfg.MODEL.HEADS.NUM_CLASSES)
        grad_logits = (logits * labels_one).sum()  # BS x num_classes
        grad_logits.backward(retain_graph=True)
        self.backbone.zero_grad()
        self.heads.zero_grad()

        # from https://github.com/1Konny/gradcam_plus_plus-pytorch/blob/master/gradcam.py
        # gradients = self.backward_features
        # activations = self.forward_features
        # weights = F.avg_pool2d(gradients, (gradients.size(-2), gradients.size(-1)))
        # mask = (weights * activations).sum(dim=1, keepdim=True)
        # mask = F.relu(mask)
        # mask = F.interpolate(mask, images.size()[2:], mode='bilinear', align_corners=False)
        # # normalizaton
        # mask_min = mask.view(mask.size(0), -1).min(1)[0]
        # mask_max = mask.view(mask.size(0), -1).max(1)[0]
        # mask = (mask - mask_min.view(-1, 1, 1, 1)).div((mask_max - mask_min + 1e-6).view(-1, 1, 1, 1)).data

        # from https://github.com/alokwhitewolf/Guided-Attention-Inference-Network/blob/master/GAIN.py
        grad = self.backward_features
        grad = F.avg_pool2d(grad, (grad.size(-2), grad.size(-1)))

        weights = self.forward_features
        Ac = []
        for i in range(grad.size(0)):
            Ac_i = F.relu(F.conv2d(weights[i].unsqueeze(0), grad[i].unsqueeze(0), None, 1, 0))
            Ac_i = (Ac_i - Ac_i.min()) / (Ac_i.max() - Ac_i.min() + 1e-6)
            Ac.append(Ac_i)
        Ac = torch.cat(Ac, dim=0)
        mask = F.interpolate(Ac, size=images.size()[2:], mode='bilinear', align_corners=False)
        # make the foreground/background more closer to 1/0.
        mask = torch.sigmoid(w * (mask - sigma))
        self.heads.train()

        return mask

    def compute_tnf_src_to_target_affine(self, target_affine, src):
        """
        :param target_affine (torch.Tensor): b, c, h, w
        :param src (torch.Tensor): b, t, c, h, w
        :return:
        corrfeat (torch.Tensor): b, t, w x h, h, w
        corrfeat_mat (torch.Tensor): b x t, w x h, h, w
        corrfeat_trans (torch.Tensor): b x t, 32
        trans_theta (torch.Tensor): b x t, 6 or 18
        """
        b, t, c, h, w = src.size()
        corrfeat = self.compute_corr_softmax(target_affine, src)  # b x t, w x h, h, w
        corrfeat_trans = self.trans_net(corrfeat).view(b * t, -1)  # b x t, 32 x 10 x 2

        trans_theta = self.linear(corrfeat_trans).contiguous().view(b * t, -1)
        trans_theta = self.transform_trans_out(trans_theta)  # b x t, 6 or 18

        return corrfeat, corrfeat_trans, trans_theta

    def compute_corr_softmax(self, target, src):
        """
        :param target: (b, c, h, w), normalized feature map
        :param src: (b, t, c, h, w), normalized feature map
        :return: corrfeat: (b x t, w x h, h, w)
        """
        b, t, c, h, w = src.size()
        base_vec = src.transpose(2, 4).contiguous().view(b, t * w * h, c)
        query_vec = target.view(b, c, h * w)

        corrfeat = torch.matmul(base_vec, query_vec) / self.temperature  # b, t * w * h, h * w
        corrfeat = corrfeat.view(b, t, w * h, h, w)
        corrfeat = F.softmax(corrfeat, dim=2).view(b * t, w * h, h, w)
        return corrfeat

    def compute_tnf_target_to_src_transfromed(self, target, src_transformed):
        """
        calculate the transformation from target to target_transformed.
        :param target: b, c, h, w
        :param src_transformed: b, t, c, h, w
        :return:
        """
        b, t, c, h, w = src_transformed.size()
        corrfeat_reverse = self.compute_corr_softmax2(target, src_transformed)  # b, t, w x h, h, w
        corrfeat_trans_reverse = self.trans_net(corrfeat_reverse).view(b * t, -1)  # b x t, 32 x 10 x 2
        trans_theta_reverse = self.linear(corrfeat_trans_reverse).contiguous().view(b * t, -1)
        trans_theta_reverse = self.transform_trans_out(trans_theta_reverse)
        return corrfeat_reverse, corrfeat_trans_reverse, trans_theta_reverse

    def compute_corr_softmax2(self, target, src_transformed):
        """
        :param target: b, c, h, w
        :param src_transformed: b, t, c, h, w
        :return:
        """
        b, t, c, h, w = src_transformed.size()
        src_transformed_vec = src_transformed.permute(0, 1, 3, 4, 2).contiguous().view(b, t * h * w, c)
        target_vec = target.transpose(2, 3).contiguous().view(b, c, w * h)

        corrfeat = torch.matmul(src_transformed_vec, target_vec) / self.temperature  # b, t * h * w, w * h
        corrfeat = F.softmax(corrfeat.view(b, t, h * w, w * h), dim=3)
        corrfeat = corrfeat.transpose(2, 3).contiguous().view(b * t, w * h, h, w)
        return corrfeat

    def transform_trans_out(self, trans_out):
        return trans_out
        # trans_out_theta = trans_out[:, 2]
        # scale = trans_out[:, 3].clamp(0, 3).unsqueeze(1)  # bound the scale
        # trans_out_2 = trans_out[:, 0].unsqueeze(1)
        # trans_out_5 = trans_out[:, 1].unsqueeze(1)
        # trans_out_0 = scale * torch.cos(trans_out_theta).unsqueeze(1)
        # trans_out_1 = scale * (- torch.sin(trans_out_theta).unsqueeze(1))
        # trans_out_3 = scale * torch.sin(trans_out_theta).unsqueeze(1)
        # trans_out_4 = scale * torch.cos(trans_out_theta).unsqueeze(1)
        # trans_out = torch.cat((trans_out_0, trans_out_1, trans_out_2,
        #                        trans_out_3, trans_out_4, trans_out_5), dim=1)
        # trans_out = trans_out.view(-1, 2, 3)

        # if self.cfg.INPUT.GEOMETRIC == 'affine':
        #     theta_aff = np.random.rand(6)
        #     translation = np.random.rand(2)
        #     alpha = (np.random.rand(1) - 0.5) * 2 * np.pi * (1.0 / 18.0)  # [-5, 5] degree
        #     theta_aff[2] = (translation[0] - 0.5) * 1 / 6.
        #     theta_aff[5] = (translation[1] - 0.5) * 1 / 6.
        #     theta_aff[0] = (1 + (theta_aff[0] - .5) * 0.2) * np.cos(alpha)
        #     theta_aff[1] = (1 + (theta_aff[1] - .5) * 0.2) * (-np.sin(alpha))
        #     theta_aff[3] = (1 + (theta_aff[3] - .5) * 0.2) * np.sin(alpha)
        #     theta_aff[4] = (1 + (theta_aff[4] - .5) * 0.2) * np.cos(alpha)
        # else:
        #     theta_aff = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1])
        #     theta_aff = theta_aff + (np.random.rand(18) - 0.5) * 2 * 0.1
        # return trans_out
