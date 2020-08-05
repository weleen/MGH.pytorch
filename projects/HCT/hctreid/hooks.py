import logging
import numpy as np

import torch
from copy import deepcopy

from fastreid.engine.train_loop import HookBase
from fastreid.engine.hooks import *
from fastreid.data.samplers import InferenceSampler
from fastreid.data.build import fast_batch_collator
from fastreid.evaluation.evaluator import inference_context


class HCTHook(HookBase):
    def __init__(self, cfg, train_set):
        super().__init__()
        self.iters_per_epoch = len(train_set) // cfg.SOLVER.IMS_PER_BATCH
        self.step_iter = cfg.HCT.EPOCHS_PER_LOOP * self.iters_per_epoch
        self.train_set = train_set
        num_train_ids = len(np.unique(np.array(self.train_set.cids)))
        self.nums_to_merge = int(num_train_ids * cfg.HCT.MERGE_PERCENT) # merge percent
        
    def before_step(self):
        # clustering
        if self.trainer.iter % self.step_iter == 0:
            ms_steps = self.trainer.cfg.HCT.MERGE_STEPS
            train_set = deepcopy(self.train_set)
            for _ in range(ms_steps):
                new_cids = self.get_new_train_data(train_set.cids, self.nums_to_merge, size_penalty=self.trainer.cfg.HCT.SIZE_PENALTY) # size_penalty
                train_set.cids = new_cids
            self.trainer.data_loader = self.trainer.build_hct_train_loader(train_set)
            self.trainer._data_loader_iter = iter(self.trainer.data_loader)
            
    def get_new_train_data(self, labels, nums_to_merge, size_penalty):
        u_feas, label_to_images = self.generate_average_feature(labels)
        dists = self.calculate_distance(u_feas)
        idx1, idx2 = self.select_merge_data(u_feas, labels, label_to_images, size_penalty, dists)
        labels = self.generate_new_train_data(idx1, idx2, labels, nums_to_merge)
        return labels

    def select_merge_data(self, u_feas, label, label_to_images,  ratio_n,  dists):
        dists.add_(torch.tril(100000 * torch.ones(len(u_feas), len(u_feas)).cuda()))

        cnt = torch.FloatTensor([len(label_to_images[label[idx]]) for idx in range(len(u_feas))]).cuda()
        dists += ratio_n * (cnt.view(1, len(cnt)) + cnt.view(len(cnt), 1))
        
        for idx in range(len(u_feas)):
            for j in range(idx + 1, len(u_feas)):
                if label[idx] == label[j]:
                    dists[idx, j] = 100000

        dists = dists.cpu().numpy()
        ind = np.unravel_index(np.argsort(dists, axis=None), dists.shape)
        idx1 = ind[0]
        idx2 = ind[1]
        return idx1, idx2

    def calculate_distance(self,u_feas):
        # calculate distance between features
        x = u_feas
        y = x
        m = len(u_feas)
        dists = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, m) + \
                torch.pow(y, 2).sum(dim=1, keepdim=True).expand(m, m).t()
        dists.addmm_(1, -2, x, y.t())
        return dists

    def generate_new_train_data(self, idx1, idx2, label, num_to_merge):
        num_before_merge = len(np.unique(np.array(label)))
        # merge clusters with minimum dissimilarity
        for i in range(len(idx1)):
            label1 = label[idx1[i]]
            label2 = label[idx2[i]]
            if label1 < label2:
                label = [label1 if x == label2 else x for x in label]
            else:
                label = [label2 if x == label1 else x for x in label]
            num_merged =  num_before_merge - len(np.sort(np.unique(np.array(label))))
            if num_merged == num_to_merge:
                break

        # set new label to the new training data
        unique_label = np.sort(np.unique(np.array(label)))
        for i in range(len(unique_label)):
            label_now = unique_label[i]
            label = [i if x == label_now else x for x in label]

        # self.train_set.cids = label

        num_after_merge = len(np.unique(np.array(label)))
        print("num of label before merge: ", num_before_merge, " after_merge: ", num_after_merge, " sub: ",
              num_before_merge - num_after_merge)
        return label
    
    def generate_average_feature(self, labels):
        #extract feature/classifier
        u_feas = self.get_feature()

        #images of the same cluster
        label_to_images = {}
        for idx, l in enumerate(labels):
            label_to_images[l] = label_to_images.get(l, []) + [idx]

        return u_feas, label_to_images
    
    def get_feature(self):
        num_workers = self.trainer.cfg.DATALOADER.NUM_WORKERS
        batch_size = self.trainer.cfg.TEST.IMS_PER_BATCH
        data_sampler = InferenceSampler(len(self.train_set))
        batch_sampler = torch.utils.data.BatchSampler(data_sampler, batch_size, False)
        dataloader = torch.utils.data.DataLoader(
            self.train_set,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=fast_batch_collator,
        )

        features = []
        model = self.trainer.model
        with inference_context(model), torch.no_grad():
            for idx, inputs in enumerate(dataloader):
                outputs = model(inputs)
                #TODO:
                features.append(outputs)
            features = torch.cat(features)

        return features