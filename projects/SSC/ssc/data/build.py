# encoding: utf-8
"""
@author:  wuyiming
@contact: yimingwu@hotmail.com
"""

import os
import torch
import numpy
from torch._six import container_abcs, string_classes, int_classes
from torch.utils.data import DataLoader
from fastreid.utils import comm

from fastreid.data import samplers
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.transforms import build_transforms
from .common import CommDataset

_root = os.getenv("FASTREID_DATASETS", "datasets")


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

    train_set = CommDataset(train_items, train_transforms, relabel=True, cfg=cfg)

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
        data_sampler = samplers.TrainingSampler(len(train_set), batch_size=cfg.SOLVER.IMS_PER_BATCH)
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

    batch_size = cfg.TEST.IMS_PER_BATCH
    data_sampler = samplers.InferenceSampler(len(test_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, batch_size, False)
    test_loader = DataLoader(
        test_set,
        batch_sampler=batch_sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
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
    elif isinstance(elem, numpy.integer):
        return torch.tensor(batched_inputs)
