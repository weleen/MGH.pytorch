# encoding: utf-8
"""
@author:  wuyiming
"""

import os
import torch
import copy
import logging

from torch.utils.data import DataLoader
from fastreid.utils import comm
from fastreid.data import samplers
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.transforms import build_transforms
from fastreid.data.build import fast_batch_collator

from .common import CommDataset

_root = os.getenv("FASTREID_DATASETS", "datasets")


def build_reid_train_loader(cfg, datasets: list = None, pseudo_labels: list = None,
                            is_train=True, relabel=True, **kwargs) -> DataLoader:
    """
    build dataloader for training and clustering.
    :param cfg: CfgNode
    :param datasets: list(ImageDataset)
    :param pseudo_labels:
    :param is_train: bool, True for training, False for clustering or validation.
    :param relabel: relabel or not
    :param kwargs:
    :return: DataLoader
    """
    cfg = cfg.clone()
    cfg.defrost()
    logger = logging.getLogger(__name__)
    dataset_names = cfg.DATASETS.NAMES

    if datasets is None:
        # generally for the first epoch
        if is_train:
            if not cfg.PSEUDO.ENABLED:
                logger.info(
                    f"The training is in a fully-supervised manner with {len(dataset_names)} dataset(s) ({dataset_names})"
                )
            else:
                no_label_datasets = [dataset_names[i] for i in cfg.PSEUDO.UNSUP]
                logger.info(
                    f"The training is in a un/semi-supervised manner with "
                    f"{len(dataset_names)} dataset(s) {dataset_names}, where {no_label_datasets} have no labels."
                )
        else:
            logger.info(
                f"Build the dataset {dataset_names} for validation or testing, where the sampler is InferenceSampler."
            )

        datasets = list()
        for d in cfg.DATASETS.NAMES:
            dataset = DATASET_REGISTRY.get(d)(root=_root, combineall=cfg.DATASETS.COMBINEALL)
            datasets.append(dataset)
    else:
        # update the datasets with given pseudo labels
        assert pseudo_labels is not None, "Please give pseudo_labels for the datasets"
        datasets = copy.deepcopy(datasets)
        for idx in cfg.PSEUDO.UNSUP:
            logger.info(f"Replace the label in dataset {dataset_names[idx]}.")
            datasets[idx].renew_labels(pseudo_labels[idx])

    train_transforms = build_transforms(cfg, is_train=is_train)
    train_set = CommDataset(datasets, train_transforms, relabel=relabel, cfg=cfg)

    iters_per_epoch = len(train_set) // cfg.SOLVER.IMS_PER_BATCH
    cfg.SOLVER.MAX_ITER *= iters_per_epoch
    num_workers = cfg.DATALOADER.NUM_WORKERS
    num_instance = cfg.DATALOADER.NUM_INSTANCE
    mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size()

    if is_train:
        data_sampler = samplers.SAMPLER_REGISTRY.get(cfg.DATALOADER.SAMPLER_NAME)(data_source=train_set.img_items,
                                                                                  batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                                                                  num_instances=num_instance,
                                                                                  size=len(train_set),
                                                                                  is_train=is_train)
    else:
        data_sampler = samplers.InferenceSampler(len(train_set))
    batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, drop_last=is_train)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
        pin_memory=True,
    )
    return train_loader


def build_reid_test_loader(cfg, dataset_name):
    cfg = cfg.clone()
    cfg.defrost()

    dataset = DATASET_REGISTRY.get(dataset_name)(root=_root)
    if comm.is_main_process():
        dataset.show_test()
    dataset.data = dataset.query + dataset.gallery

    test_transforms = build_transforms(cfg, is_train=False)
    test_set = CommDataset(dataset, test_transforms, relabel=False)

    mini_batch_size = cfg.TEST.IMS_PER_BATCH // comm.get_world_size()
    data_sampler = samplers.InferenceSampler(len(test_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, mini_batch_size, False)
    test_loader = DataLoader(
        test_set,
        batch_sampler=batch_sampler,
        num_workers=0,  # save some memory
        collate_fn=fast_batch_collator,
        pin_memory=True,
    )
    return test_loader, len(dataset.query)
