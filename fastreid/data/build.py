# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import os

import torch
import numpy
import logging
from torch._six import container_abcs, string_classes, int_classes
from torch.utils.data import DataLoader

from fastreid.utils import comm
from . import samplers
from .common import CommDataset
from .datasets import DATASET_REGISTRY
from .transforms import build_transforms

_root = os.getenv("FASTREID_DATASETS", "datasets")
logger = logging.getLogger(__name__)


def build_reid_train_loader(cfg, datasets: list = None, pseudo_labels: list = None, 
                            is_train=True, relabel=True, for_clustering=False, mapper=None, **kwargs) -> DataLoader:
    cfg = cfg.clone()
    dataset_names = cfg.DATASETS.NAMES
    if for_clustering:
        assert is_train is False, "is_train should be False for clustering."
        dataset_names = tuple([dataset_names[idx] for idx in cfg.PSEUDO.UNSUP])

    if datasets is None:
        # Generally for the first epoch, the datasets have not been built.
        if is_train:
            if not cfg.PSEUDO.ENABLED:
                logger.info(f"The training is in a fully-supervised manner with {len(dataset_names)} dataset(s) ({dataset_names})")
            else:
                no_label_datasets = [dataset_names[i] for i in cfg.PSEUDO.UNSUP]
                logger.info(f"The training is in a un/semi-supervised manner with {len(dataset_names)} dataset(s) {dataset_names}, where {no_label_datasets} have no labels.")
        else:
            logger.info(f"Build the dataset {dataset_names} for extracting features, where the sampler is InferenceSampler.")

        datasets = list()
        # Build all datasets with groundtruth labels.
        for d in dataset_names:
            dataset = DATASET_REGISTRY.get(d)(root=_root, combineall=cfg.DATASETS.COMBINEALL, mode='train', **kwargs)
            if comm.is_main_process():
                dataset.show_train()
            datasets.append(dataset)
    else:
        assert pseudo_labels is not None, "Please give pseudo_labels for the datasets."
        for idx, dataset in enumerate(datasets):
            logger.info(f"Replace the unlabeled label in dataset {dataset_names[idx]} with pseudo labels.")
            datasets[idx].renew_labels(pseudo_labels[idx])
    if mapper is not None:
        transforms = mapper
    else:
        if for_clustering and cfg.PSEUDO.CLUSTER_AUG:
            # when extract feature in clustering, use augmentation to extract robust features.
            frozen = cfg.is_frozen()
            cfg.defrost()
            is_mutual = cfg.INPUT.MUTUAL.ENABLED
            cfg.INPUT.MUTUAL.ENABLED = True
            transforms = build_transforms(cfg, is_train=True)
            cfg.INPUT.MUTUAL.ENABLED = is_mutual
            if frozen: cfg.freeze()
        else:
            transforms = build_transforms(cfg, is_train=is_train)
    train_set = CommDataset(datasets, transforms, relabel=relabel)  # relabel image pid and camid when relabel = True

    num_workers = cfg.DATALOADER.NUM_WORKERS // comm.get_world_size()
    num_instance = cfg.DATALOADER.NUM_INSTANCE
    mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size()

    if is_train:
        data_sampler = samplers.SAMPLER_REGISTRY.get(cfg.DATALOADER.SAMPLER_NAME)(data_source=train_set.img_items,
                                                                                  batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                                                                  num_instances=num_instance,
                                                                                  size=len(train_set),
                                                                                  train_set=train_set)
    else:
        data_sampler = samplers.InferenceSampler(len(train_set))
    batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, drop_last=is_train)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        # collate_fn=fast_batch_collator,
        pin_memory=True,
    )
    return train_loader


def build_reid_test_loader(cfg, dataset_name, mode='test', mapper=None, **kwargs):
    cfg = cfg.clone()

    dataset = DATASET_REGISTRY.get(dataset_name)(root=_root, mode=mode, **kwargs)
    if comm.is_main_process():
        dataset.show_test()

    if mapper is not None:
        transforms = mapper
    else:
        transforms = build_transforms(cfg, is_train=False)
    test_set = CommDataset(dataset, transforms, relabel=False)

    mini_batch_size = cfg.TEST.IMS_PER_BATCH // comm.get_world_size()
    num_workers = cfg.DATALOADER.NUM_WORKERS // comm.get_world_size()
    data_sampler = samplers.InferenceSampler(len(test_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, mini_batch_size, False)
    test_loader = DataLoader(
        test_set,
        batch_sampler=batch_sampler,
        num_workers=num_workers,  # save some memory
        collate_fn=fast_batch_collator,
        pin_memory=True,
    )
    return test_loader, len(dataset.query)


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def fast_batch_collator(batched_inputs):
    """
    A simple batch collator for most common reid tasks
    """
    elem = batched_inputs[0]
    if isinstance(elem, torch.Tensor):
        out = torch.zeros((len(batched_inputs), *elem.size()), dtype=elem.dtype)
        for i, tensor in enumerate(batched_inputs):
            out[i] += tensor
        return out

    elif isinstance(elem, container_abcs.Mapping):
        return {key: fast_batch_collator([d[key] for d in batched_inputs]) for key in elem}

    elif isinstance(elem, float):
        return torch.tensor(batched_inputs, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batched_inputs)
    elif isinstance(elem, string_classes):
        return batched_inputs
    elif isinstance(elem, numpy.integer):
        return torch.tensor(batched_inputs)
