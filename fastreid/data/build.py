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
from .common import CommDataset
from .datasets import DATASET_REGISTRY
from .transforms import build_transforms

_root = os.getenv("FASTREID_DATASETS", "datasets")


def build_reid_train_loader(cfg, datasets: list = None, pseudo_labels: list = None, cam_labels: list = None,
                            is_train=True, relabel=True, for_clustering=False, **kwargs) -> DataLoader:
    """
    build dataloader for training and clustering.
    :param cfg(CfgNode): config
    :param datasets(list(ImageDataset)): dataset information, include img_path, pid, camid.
    :param pseudo_labels(list): generated pseudo labels for un/semi-supervised learning.
    :param cam_labels(list): camera id for all datasets.
    :param is_train(bool): True for training, the sampler and transformer are TrainSampler and train_transformer. False for clustering, the sampler and transformer are InferenceSampler and test_transformer.
    :param relabel(bool): relabel or not.
    :param for_clustering(bool): True means building dataloader for clustering, the labeled dataset is ignored.
    :param kwargs:
    :return: DataLoader
    """
    cfg = cfg.clone()
    cfg.defrost()
    logger = logging.getLogger(__name__)
    dataset_names = cfg.DATASETS.NAMES
    if for_clustering:
        dataset_names = [dataset_names[idx] for idx in cfg.PSEUDO.UNSUP]

    if datasets is None:
        # Generally for the first epoch, the datasets have not been built.
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
                f"Build the dataset {dataset_names} for extracting features, where the sampler is InferenceSampler."
            )

        datasets = list()
        # Build all datasets with groundtruth labels.
        for d in dataset_names:
            dataset = DATASET_REGISTRY.get(d)(root=_root, combineall=cfg.DATASETS.COMBINEALL)
            datasets.append(dataset)
    else:
        # update unlabeled datasets with given pseudo labels, 
        # update labeled datasets if label is string with dataset_name and relabel should be False.
        assert pseudo_labels is not None, "Please give pseudo_labels for the datasets."
        assert relabel is False, "Please set relabel to False when using pseudo labels."
        datasets = copy.deepcopy(datasets)
        for idx, dataset in enumerate(datasets):
            if idx in cfg.PSEUDO.UNSUP:
                logger.info(f"Replace the unlabeled label in dataset {dataset_names[idx]} with pseudo labels.")
                datasets[idx].renew_labels(pseudo_labels[idx], cam_labels[idx])
            else:
                if isinstance(dataset.data[0][1], str):
                    logger.warning(f"Replace the string label in labeled dataset {dataset_names[idx]} with int label. Only on the first iteration")
                    datasets[idx].renew_labels(pseudo_labels[idx], cam_labels[idx])
    # TODO: multiple datasets training in unsupervised domain adaptation is not supported now.
    # Please refer to implementation of OpenUnReID.
    train_transforms = build_transforms(cfg, is_train=is_train)
    train_set = CommDataset(datasets, train_transforms, relabel=relabel)

    iters_per_epoch = len(train_set) // cfg.SOLVER.IMS_PER_BATCH
    cfg.SOLVER.MAX_ITER *= iters_per_epoch
    num_workers = cfg.DATALOADER.NUM_WORKERS
    num_instance = cfg.DATALOADER.NUM_INSTANCE
    mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size()

    if is_train:
        data_sampler = samplers.SAMPLER_REGISTRY.get(cfg.DATALOADER.SAMPLER_NAME)(data_source=train_set.img_items,
                                                                                  batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                                                                  num_instances=num_instance,
                                                                                  size=len(train_set))
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


def build_reid_test_loader(cfg, dataset_name, val=False):
    """
    cfg (CfgNode): configs.
    dataset_name (str): name of the dataset.
    val (bool): run validation or testing.
    """
    cfg = cfg.clone()
    cfg.defrost()

    dataset = DATASET_REGISTRY.get(dataset_name)(root=_root, val=val)
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
