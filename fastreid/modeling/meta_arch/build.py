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


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    logger = logging.getLogger(__name__)
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))

    frozen = cfg.is_frozen()
    cfg.defrost()

    # convert domain-specific batch normalization
    num_domains = len(cfg.DATASETS.NAMES)
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

    # convert to sync bn (optional)
    rank, world_size = comm.get_rank(), comm.get_world_size()
    sync_bn = (cfg.MODEL.BACKBONE.NORM == cfg.MODEL.HEADS.NORM == 'syncBN')
    samples_per_gpu = cfg.SOLVER.IMS_PER_BATCH // world_size
    if sync_bn and world_size > 1:
        if samples_per_gpu < cfg.MODEL.SAMPLES_PER_BN:
            total_batch_size = cfg.SOLVER.IMS_PER_BATCH
            if total_batch_size > cfg.MODEL.SAMPLES_PER_BN:
                assert total_batch_size % cfg.MODEL.SAMPLES_PER_BN == 0, "Samples for sync_bn cannot be evenly divided."
                group_num = int(total_batch_size // cfg.MODEL.SAMPLES_PER_BN)
                dist_groups = comm.simple_group_split(world_size, rank, group_num)
            else:
                dist_groups = None
                logger.warning(
                    f"'Dist_group' is switched off, since samples_per_bn "
                    f"({cfg.MODEL.SAMPLES_PER_BN}) is larger than or equal to "
                    f"total_batch_size ({total_batch_size})."
                )
            convert_sync_bn(model, dist_groups)
        else:
            logger.warning(
                f"Sync BN is switched off, since samples ({cfg.MODEL.SAMPLES_PER_BN})"
                f" per BN are fewer than or same as samples {samples_per_gpu}) per GPU."
            )
    else:
        logger.warning(
            "Sync BN is switched off, since the program is running without DDP"
        )
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
