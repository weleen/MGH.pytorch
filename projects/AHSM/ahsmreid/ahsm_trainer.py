# encoding: utf-8
"""
@author:  wenhuzhang
@contact: Andrew-pph@outlook.com
"""
import logging
import time
import datetime
from collections import OrderedDict

from typing import List

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel, DataParallel
from fvcore.common.config import CfgNode
from fvcore.common.checkpoint import Checkpointer
from torch.utils.data.sampler import SubsetRandomSampler

from fastreid.data.build import DATASET_REGISTRY, fast_batch_collator

from fastreid.data.transforms import build_transforms
from fastreid.engine.defaults import DefaultTrainer
from fastreid.layers.sync_bn import patch_replication_callback
from fastreid.utils.logger import setup_logger, log_every_n_seconds
from fastreid.utils import comm
from fastreid.evaluation.evaluator import inference_context

from fastreid.engine import hooks
from fastreid.data.common import CommDataset

import random
from fastreid.data import samplers
__all__ = ["AHSMTrainer"]


class AHSMTrainer(DefaultTrainer):
    def __init__(self, cfg: CfgNode) -> None:
        self.cfg = cfg
        logger = logging.getLogger('fastreid.' + __name__)
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()
        # build model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        logger.info('Prepare active hard sampling data set')
        data_loader,self.labeled_set,self.unlabeled_set = self.build_active_sample_dataloader(is_train=False)       
        self.label_num = len(self.labeled_set)
        # For training, wrap with DP. But don't need this for inference.
        model = DataParallel(model)
        if cfg.MODEL.BACKBONE.NORM == "syncBN":
            # Monkey-patching with syncBN
            patch_replication_callback(model)
        model = model.cuda()
        super(DefaultTrainer, self).__init__(model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = Checkpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        
        
        self.max_iter = cfg.ACTIVE.TRAIN_CYCLES

        #self.register_hooks(self.build_hooks())

    def build_active_sample_dataloader(self,
                                datalist: List = None,
                                is_train: bool = False,
                                is_sample_loader:bool=False
                                ) -> torch.utils.data.DataLoader:
        """
        :param datalist: dataset list. if dataset is None, random initialize the labeled/unlabeled list.
        :param is_train: build training transformation and sampler.
        :return:
        """
        transforms = build_transforms(self.cfg, is_train=is_train)
        train_items = list()
        for d in self.cfg.DATASETS.NAMES:
            dataset = DATASET_REGISTRY.get(d)(combineall=self.cfg.DATASETS.COMBINEALL)
            dataset.show_train()
            train_items.extend(dataset.train)

        if datalist is None:
            indices = list(range(len(train_items)))
            random.shuffle(indices)
            random_num=int(self.cfg.ACTIVE.INITIAL_RATE * len(train_items)/2)*2 #must be even number
            labeled_set = indices[:random_num]
            unlabeled_set = indices[random_num:]
            load_items = labeled_set
        else:
            load_items = datalist  
        
        data_set = CommDataset([train_items[i] for i in load_items], transforms, relabel=True)
        print('Length of the labeled dataset:',data_set.__len__())
        num_workers = self.cfg.DATALOADER.NUM_WORKERS
        batch_size = self.cfg.SOLVER.IMS_PER_BATCH
        num_instance = self.cfg.DATALOADER.NUM_INSTANCE

        
        if self.cfg.DATALOADER.PK_SAMPLER:
            data_sampler = samplers.RandomIdentitySampler(data_set.img_items, batch_size, num_instance)
        else:
            data_sampler = samplers.TrainingSampler(len(data_set))
        batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, batch_size, True)

        data_loader = torch.utils.data.DataLoader(
            data_set,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=fast_batch_collator,
        )
        if datalist is None:
            return data_loader,labeled_set,unlabeled_set
        else:
            return data_loader

   
    

    def active_train(self):
        for cycle in range(self.cfg.ACTIVE.SAMPLE_CYCLES):
            #print('!!!!!!!:',cycle,len(self.labeled_set),len(self.unlabeled_set))
            self.register_hooks(self.build_hooks())
            DefaultTrainer.train(self)
            self.labeled_set,self.unlabeled_set=\
                    self.random_sample(self.labeled_set,self.unlabeled_set,self.label_num)
            #unlabeled_loader=self.build_active_sample_dataloader(self.unlabeled_set,is_sample_loader=True)           
            self.data_loader=self.build_active_sample_dataloader(self.labeled_set)
            

    def random_sample(self,labeled_set,unlabeled_set,labeled_num):
        new_labeled_set = labeled_set
        new_unlabeled_set = unlabeled_set
        random.shuffle(new_unlabeled_set)

        new_labeled_set+=new_unlabeled_set[:labeled_num]
        new_unlabeled_set=new_unlabeled_set[labeled_num:]
        return new_labeled_set,new_unlabeled_set

    def uncertainty_sample(self,model,unlabeled_loader):

        

        return unlabeled_loader


