# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import os
import torch
import copy
import numpy
import logging
from torch._six import container_abcs, string_classes, int_classes
from torch.utils.data import DataLoader
from fastreid.utils import comm

from . import samplers
from .common import CommDataset, NewCommDataset
from .datasets import DATASET_REGISTRY
from .transforms import build_transforms

_root = os.getenv("FASTREID_DATASETS", "datasets")


def build_reid_train_loader(cfg):
    cfg = cfg.clone()
    cfg.defrost()

    train_items = list()
    for d in cfg.DATASETS.NAMES:
        dataset = DATASET_REGISTRY.get(d)(root=_root, combineall=cfg.DATASETS.COMBINEALL)
        if comm.is_main_process():
            dataset.show_train()
        train_items.extend(dataset.train)

    iters_per_epoch = len(train_items) // cfg.SOLVER.IMS_PER_BATCH
    cfg.SOLVER.MAX_ITER *= iters_per_epoch
    train_transforms = build_transforms(cfg, is_train=True)
    train_set = CommDataset(train_items, train_transforms, relabel=True)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    num_instance = cfg.DATALOADER.NUM_INSTANCE
    mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size()

    if cfg.DATALOADER.PK_SAMPLER:
        if cfg.DATALOADER.NAIVE_WAY:
            data_sampler = samplers.NaiveIdentitySampler(train_set.img_items,
                                                         cfg.SOLVER.IMS_PER_BATCH, num_instance)
        else:
            data_sampler = samplers.BalancedIdentitySampler(train_set.img_items,
                                                            cfg.SOLVER.IMS_PER_BATCH, num_instance)
    else:
        data_sampler = samplers.TrainingSampler(len(train_set))
    batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
    )
    return train_loader


def build_reid_test_loader(cfg, dataset_name):
    cfg = cfg.clone()
    cfg.defrost()

    dataset = DATASET_REGISTRY.get(dataset_name)(root=_root)
    if comm.is_main_process():
        dataset.show_test()
    test_items = dataset.query + dataset.gallery

    test_transforms = build_transforms(cfg, is_train=False)
    test_set = CommDataset(test_items, test_transforms, relabel=False)

    mini_batch_size = cfg.TEST.IMS_PER_BATCH // comm.get_world_size()
    data_sampler = samplers.InferenceSampler(len(test_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, mini_batch_size, False)
    test_loader = DataLoader(
        test_set,
        batch_sampler=batch_sampler,
        num_workers=0,  # save some memory
        collate_fn=fast_batch_collator)
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


def build_reid_train_loader_new(cfg, datasets: list = None, pseudo_labels: list = None,
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
        for idx, d in enumerate(cfg.DATASETS.NAMES):
            dataset = DATASET_REGISTRY.get(d)(root=_root, combineall=cfg.DATASETS.COMBINEALL,
                                              cuhk03_labeled=cfg.DATASETS.CUHK03.LABELED,
                                              cuhk03_classic_split=cfg.DATASETS.CUHK03.CLASSIC_SPLIT)
            datasets.append(dataset)
    else:
        # update the datasets with given pseudo labels
        assert pseudo_labels is not None, "Please give pseudo_labels for the datasets"
        datasets = copy.deepcopy(datasets)
        for idx in cfg.PSEUDO.UNSUP:
            logger.info(f"Replace the label in dataset {dataset_names[idx]}.")
            datasets[idx].renew_labels(pseudo_labels[idx])

    train_transforms = build_transforms(cfg, is_train=is_train)
    train_set = NewCommDataset(datasets, train_transforms, relabel=relabel)

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
        pin_memory=True
    )
    return train_loader


def build_reid_test_loader_new(cfg, dataset_name):
    """
    build data loader for testing
    :param cfg: CfgNode
    :param dataset_name: str
    :return: DataLoader
    """
    cfg = cfg.clone()
    cfg.defrost()

    dataset = DATASET_REGISTRY.get(dataset_name)(root=_root,
                                                 cuhk03_labeled=cfg.DATASETS.CUHK03.LABELED,
                                                 cuhk03_classic_split=cfg.DATASETS.CUHK03.CLASSIC_SPLIT,
                                                 market1501_500k=cfg.DATASETS.MARKET1501.ENABLE_500K)
    if comm.is_main_process():
        dataset.show_test()
    # TODO: this code is tricky, combine query and gallery.
    dataset.data = dataset.query + dataset.gallery

    test_transforms = build_transforms(cfg, is_train=False)
    test_set = NewCommDataset(dataset, test_transforms, relabel=False)

    mini_batch_size = cfg.TEST.IMS_PER_BATCH // comm.get_world_size()
    data_sampler = samplers.InferenceSampler(len(test_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, mini_batch_size, False)
    test_loader = DataLoader(
        test_set,
        batch_sampler=batch_sampler,
        num_workers=0,  # save some memory
        collate_fn=fast_batch_collator,
        pin_memory=True)
    return test_loader, len(dataset.query)
