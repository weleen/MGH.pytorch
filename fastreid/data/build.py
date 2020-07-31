# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import os
import logging
import torch
from torch._six import container_abcs, string_classes, int_classes
from torch.utils.data import DataLoader
from fastreid.utils import comm

from . import samplers
from .common import CommDataset, NewCommDataset
from .datasets import DATASET_REGISTRY
from .samplers import SAMPLER_REGISTRY
from .transforms import build_transforms

_root = os.getenv("FASTREID_DATASETS", "datasets")


def build_reid_train_loader_new(cfg, train_items=None, pseudo_labels=None, epoch=0, **kwargs):
    """
    build the dataloader
    :param cfg:
    :param train_items:
    :param pseudo_labels:
    :param epoch:
    :param kwargs:
    :return:
    """
    logger = logging.getLogger(__name__)
    dataset_names = cfg.DATASETS.NAMES
    dataset_modes = cfg.DATASETS.TRAINS_MODE
    unsup_dataset_indexes = cfg.DATASETS.TRAINS_UNSUPERVISED
    assert len(dataset_names) == len(dataset_modes), "names and modes of the datasets" \
                                                     " is not match, ({} vs. {})".format(dataset_names, dataset_modes)

    for mode in dataset_modes:
        assert mode in [
            "train",
            "trainval",
        ], "subset for training should be selected in [train, trainval]"

    if train_items is None:
        # generally for the first epoch
        if len(unsup_dataset_indexes) == 0:
            logger.info(
                f"The training is in a fully-supervised manner with "
                f"{len(dataset_names)} dataset(s) ({dataset_names})"
            )
        else:
            no_label_datasets = [dataset_names[i] for i in unsup_dataset_indexes]
            logger.info(
                f"The training is in a un/semi-supervised manner with "
                f"{len(dataset_names)} dataset(s) {dataset_names},\n"
                f"where {no_label_datasets} have no labels."
            )

        train_items = list()

        for idx, (d, dm) in enumerate(zip(cfg.DATASETS.NAMES, cfg.DATASETS.TRAINS_MODE)):
            dataset = DATASET_REGISTRY.get(d)(root=_root, combineall=cfg.DATASETS.COMBINEALL,
                                              cuhk03_labeled=cfg.DATASETS.CUHK03.LABELED,
                                              cuhk03_classic_split=cfg.DATASETS.CUHK03.CLASSIC_SPLIT)
            if idx in unsup_dataset_indexes:
                # remove the labels in unsupervised dataset
                try:
                    new_labels = pseudo_labels[idx]
                except:
                    new_labels = list(range(len(dataset)))
                    logger.warning(f"no labels are provided for {d}")
                dataset.renew_labels(new_labels)
            if comm.is_main_process():
                dataset.show_train()
            train_items.append(getattr(dataset, dm))
    else:
        for i, idx in enumerate(unsup_dataset_indexes):
            train_items[idx].renew_labels(pseudo_labels[i])

    train_transforms = build_transforms(cfg, is_train=True)
    train_set = NewCommDataset(train_items, train_transforms, relabel=True)

    num_workers = cfg.DATALOADER.NUM_WORKERS // comm.get_world_size()
    num_instance = cfg.DATALOADER.NUM_INSTANCE
    mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size()

    data_sampler = SAMPLER_REGISTRY.get(cfg.DATALOADER.SAMPLER_NAME)(data_source=train_set.dataset,
                                                                     batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                                                     num_instances=num_instance,
                                                                     size=len(train_set))
    batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
    )
    return train_loader


def build_reid_val_loader_new(cfg):
    test_transforms = build_transforms(cfg, is_train=False)
    dataset_name = cfg.DATASETS.VAL
    dataset = DATASET_REGISTRY.get(dataset_name)(root=_root,
                                                 cuhk03_labeled=cfg.DATASETS.CUHK03.LABELED,
                                                 cuhk03_classic_split=cfg.DATASETS.CUHK03.CLASSIC_SPLIT,
                                                 market1501_500k=cfg.DATASETS.MARKET1501.ENABLE_500K)
    if comm.is_main_process():
        dataset.show_val()
    val_items = dataset.val

    val_set = CommDataset(val_items, test_transforms, relabel=False)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    batch_size = cfg.TEST.IMS_PER_BATCH
    data_sampler = samplers.InferenceSampler(len(val_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, batch_size, False)
    val_loader = DataLoader(
        val_set,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=fast_batch_collator)
    return val_loader


def build_reid_train_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)

    train_items = list()
    for d in cfg.DATASETS.NAMES:
        dataset = DATASET_REGISTRY.get(d)(root=_root, combineall=cfg.DATASETS.COMBINEALL,
                                          cuhk03_labeled=cfg.DATASETS.CUHK03.LABELED,
                                          cuhk03_classic_split=cfg.DATASETS.CUHK03.CLASSIC_SPLIT)
        if comm.is_main_process():
            dataset.show_train()
        train_items.extend(dataset.train)

    train_set = CommDataset(train_items, train_transforms, relabel=True)

    num_workers = cfg.DATALOADER.NUM_WORKERS // comm.get_world_size()
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
    test_transforms = build_transforms(cfg, is_train=False)

    dataset = DATASET_REGISTRY.get(dataset_name)(root=_root,
                                                 cuhk03_labeled=cfg.DATASETS.CUHK03.LABELED,
                                                 cuhk03_classic_split=cfg.DATASETS.CUHK03.CLASSIC_SPLIT,
                                                 market1501_500k=cfg.DATASETS.MARKET1501.ENABLE_500K)
    if comm.is_main_process():
        dataset.show_test()
    test_items = dataset.query + dataset.gallery

    test_set = CommDataset(test_items, test_transforms, relabel=False)

    num_workers = cfg.DATALOADER.NUM_WORKERS // comm.get_world_size()
    batch_size = cfg.TEST.IMS_PER_BATCH
    data_sampler = samplers.InferenceSampler(len(test_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, batch_size, False)
    test_loader = DataLoader(
        test_set,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=fast_batch_collator)
    return test_loader, len(dataset.query)


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
