# encoding: utf-8
"""
@author:  xingyu liao
@contact: liaoxingyu5@jd.com
"""

from .. import losses as Loss


def reid_losses(cfg, outs: dict, inputs: dict, prefix='', **kwargs) -> dict:
    outputs            = outs["outputs"]
    gt_classes         = outs["targets"]

    loss_dict = {}
    for loss_name in cfg.MODEL.LOSSES.NAME:
        if loss_name == 'ContrastiveLoss':
            pred_features = outputs['features']
            loss = {"contrastive_loss": kwargs['memory'](pred_features, inputs['index'])}
        else:
            cls_outputs = outputs['cls_outputs']
            pred_features = outputs['features']
            if cfg.PSEUDO.ENABLED and cfg.PSEUDO.WITH_CLASSIFIER:
                cls_outputs = cls_outputs[:, :cfg.MODEL.HEADS.NUM_CLASSES]
            loss = getattr(Loss, loss_name)(cfg)(cls_outputs, pred_features, gt_classes)
        loss_dict.update(loss)
    # rename
    named_loss_dict = {}
    for name in loss_dict.keys():
        named_loss_dict[prefix + name] = loss_dict[name]
    del loss_dict
    return named_loss_dict
