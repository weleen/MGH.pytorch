# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import torch
import logging

from fvcore.common.registry import Registry
from fastreid.layers import convert_dsbn, convert_sync_bn
from fastreid.utils import comm

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.
The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_model(cfg, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    logger = logging.getLogger(__name__)
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg, **kwargs)
    model.to(torch.device(cfg.MODEL.DEVICE))

    frozen = cfg.is_frozen()
    cfg.defrost()

    # convert domain-specific batch normalization
    num_domains = len(cfg.DATASETS.NAMES)
    if num_domains > 1:
        # TODO: multiple datasets are not supported in un/semi-supervised learning.
        assert not cfg.PSEUDO.ENABLED, "multiple datasets are not supported in un/semi-supervised learning."
    if num_domains > 1 and cfg.MODEL.DSBN:
        # TODO: set the first test dataset as the target dataset.
        target_dataset = cfg.DATASETS.TESTS[0]
        if target_dataset in cfg.DATASETS.NAMES:
            target_domain_index = cfg.DATASETS.NAMES.index(target_dataset)
        else:
            target_domain_index = -1
            logger.warning(
                f"the domain of {target_dataset} for testing is not within "
                f"train sets, we use {cfg.DATASETS.NAMES[-1]}'s BN intead, "
                f"which may cause unsatisfied performance.")
        convert_dsbn(model, num_domains, target_domain_index)
    else:
        logger.warning(
            "domain-specific BN is switched off, since there's only one domain."
        )
        cfg.MODEL.DSBN = False

    # create mean teacher network (optional)
    if cfg.MODEL.MEAN_NET:
        model = TeacherStudentNetwork(model, cfg.MODEL.MEAN_NET_ALPHA)

    if frozen: cfg.freeze()
    
    return model


class TeacherStudentNetwork(torch.nn.Module):
    """
    TeacherStudentNetwork.
    """
    def __init__(self, net, alpha=0.999):
        super(TeacherStudentNetwork, self).__init__()
        self.net = net
        self.mean_net = copy.deepcopy(self.net)

        for param, param_m in zip(self.net.parameters(), self.mean_net.parameters()):
            param_m.data.copy_(param.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        self.alpha = alpha

    def forward(self, x):
        if not self.training:
            return self.mean_net(x)

        results = self.net(x)

        with torch.no_grad():
            self._update_mean_net()  # update mean net
            results_m = self.mean_net(x)

        return results, results_m

    @torch.no_grad()
    def initialize_centers(self, centers, labels):
        self.net.initialize_centers(centers, labels)
        self.mean_net.initialize_centers(centers, labels)

    @torch.no_grad()
    def _update_mean_net(self):
        for param, param_m in zip(self.net.parameters(), self.mean_net.parameters()):
            param_m.data.mul_(self.alpha).add_(param.data, alpha=1-self.alpha)
