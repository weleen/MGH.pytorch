import numpy as np
import torch

from fastreid.data.build import fast_batch_collator
from fastreid.data.samplers import InferenceSampler
from fastreid.engine.train_loop import HookBase
from fastreid.evaluation.evaluator import inference_context
from .exloss import ExLoss


class BUCHook(HookBase):
    def __init__(self, cfg, train_set):
        super().__init__()
        self.iters_per_epoch = len(train_set) // cfg.SOLVER.IMS_PER_BATCH
        self.loops = 0
        self.step_iter = (2 * self.loops + 20) * self.iters_per_epoch
        self.train_set = train_set
        num_train_ids = len(np.unique(np.array(self.train_set.cids)))
        self.nums_to_merge = int(num_train_ids * 0.05)  # merge percent

    def after_step(self):
        # clustering
        if self.trainer.iter > 0 and self.trainer.iter % self.step_iter == 0:
            new_cids = self.get_new_train_data(self.train_set.cids, self.nums_to_merge,
                                               size_penalty=0.003)  # size_penalty
            self.train_set.cids = new_cids
            self.trainer.data_loader = self.trainer.build_buc_train_loader(self.train_set)
            self.trainer._data_loader_iter = iter(self.trainer.data_loader)
            self.loops += 1
            self.step_iter = (2 * self.loops + 20) * self.iters_per_epoch
            num_train_ids = len(np.unique(np.array(new_cids)))
            self.trainer.criterion = ExLoss(num_train_ids, num_features=2048, t=10).cuda()

    def get_new_train_data(self, labels, nums_to_merge, size_penalty):
        u_feas, label_to_images = self.generate_average_feature(labels)
        dists = self.calculate_distance(u_feas)
        idx1, idx2 = self.select_merge_data(u_feas, labels, label_to_images, size_penalty, dists)
        labels = self.generate_new_train_data(idx1, idx2, labels, nums_to_merge)
        return labels

    def select_merge_data(self, u_feas, label, label_to_images, ratio_n, dists):
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

    def calculate_distance(self, u_feas):
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
            num_merged = num_before_merge - len(np.sort(np.unique(np.array(label))))
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
        # extract feature/classifier
        u_feas = self.get_feature()

        # images of the same cluster
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
                # TODO:
                features.append(outputs)
            features = torch.cat(features)

        return features
