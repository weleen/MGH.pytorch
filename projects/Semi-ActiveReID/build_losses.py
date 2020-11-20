'''
Author: WuYiming
Date: 2020-10-23 22:55:55
LastEditTime: 2020-11-17 10:05:26
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /fast-reid/projects/ActiveReID/build_losses.py
'''
import fastreid.modeling.losses as Loss


def reid_losses(cfg, outs: dict, inputs: dict, prefix='', **kwargs) -> dict:
    outputs            = outs["outputs"]
    gt_classes         = outs["targets"]

    loss_dict = {}
    for loss_name in cfg.MODEL.LOSSES.NAME:
        if loss_name == 'ContrastiveLoss':
            pred_features = outputs['features']
            if 'contrastive_loss_weight' in kwargs:
                loss = {"contrastive_loss": kwargs['contrastive_loss_weight'] * kwargs['memory'](pred_features, inputs['index'], **kwargs)}
            else:
                loss = {"contrastive_loss": kwargs['memory'](pred_features, inputs['index'], **kwargs)}
        else:
            cls_outputs = outputs['cls_outputs']
            if loss_name == 'TripletLoss':
                pred_features = outputs['before_features']
            else:
                pred_features = outputs['features']
            if cfg.PSEUDO.ENABLED and cfg.PSEUDO.WITH_CLASSIFIER:
                cls_outputs = cls_outputs[:, :cfg.MODEL.HEADS.NUM_CLASSES]
            loss = getattr(Loss, loss_name)(cfg)(cls_outputs, pred_features, gt_classes)
            if loss_name == 'ActiveTripletLoss' and 'active_triplet_loss_weight' in kwargs:
                loss['loss_triplet_a'] = loss['loss_triplet_a'] * kwargs['active_triplet_loss_weight']
        loss_dict.update(loss)
    # rename
    named_loss_dict = {}
    for name in loss_dict.keys():
        named_loss_dict[prefix + name] = loss_dict[name]
    del loss_dict
    return named_loss_dict