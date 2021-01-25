# encoding: utf-8
"""
@author:  xingyu liao
@contact: liaoxingyu5@jd.com
"""

from .. import losses as Loss


def reid_losses(cfg, outs: dict, inputs: dict=None, prefix='', **kwargs) -> dict:
    outputs            = outs["outputs"]
    gt_classes         = outs["targets"]

    loss_dict = {}
    for loss_name in cfg.MODEL.LOSSES.NAME:
        if loss_name == 'ContrastiveLoss':
            assert 'memory' in kwargs, 'memory is not in kwargs.'
            pred_features = outputs['features']
            loss = {"contrastive_loss": kwargs['memory'](pred_features, inputs['index'], **kwargs)}
        else:
            cls_outputs = outputs['cls_outputs']
            if loss_name == 'TripletLoss' and 'before_features' in outputs:
                pred_features = outputs['before_features']
            elif loss_name == 'PairwiseSmoothLoss':
                assert 'low_level_feature' in outs, 'no low-level feature in output for PairwiseSmoothLoss'
                pred_features = outs['low_level_feature']
            else:
                pred_features = outputs['features']
            if cfg.PSEUDO.ENABLED and cfg.PSEUDO.WITH_CLASSIFIER:
                cls_outputs = cls_outputs[:, :cfg.MODEL.HEADS.NUM_CLASSES]
            loss = getattr(Loss, loss_name)(cfg)(cls_outputs, pred_features, gt_classes, **kwargs)
        loss_dict.update(loss)
    # rename
    named_loss_dict = {}
    for name in loss_dict.keys():
        named_loss_dict[prefix + name] = loss_dict[name]
    del loss_dict
    return named_loss_dict
