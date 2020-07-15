from fastreid.engine.train_loop import HookBase
from fastreid.engine.hooks import *
from fastreid.data.samplers import InferenceSampler
from fastreid.data.build import fast_batch_collator
from fastreid.evaluation.evaluator import inference_context
from samplers import build_active_samplers
import torch


class DataloaderHook(HookBase):
    def __init__(self, cfg, data_len):
        super().__init__()
        self.base_iter = cfg.ACTIVE.TRAIN_CYCLES
        labeled_num = int(data_len * cfg.ACTIVE.INITIAL_RATE) + 1
        self.samplers = build_active_samplers(cfg)
        index_set = [i for i in range(data_len)]
        index_dataloader = torch.utils.data.DataLoader(index_set, batch_size=labeled_num, shuffle=True)
        self._index_iter = iter(index_dataloader)
        self.sample_iter = self.base_iter

    def before_step(self):
        if self.trainer.iter % self.sample_iter == 0:
            indexes = self._index_iter.next()
            features, targets = self.get_feature()
            sel_feats = features[indexes]
            dist_mat = self.euclidean_dist(sel_feats, features)
            # only choose first 30 similar instances
            sim_mat = torch.argsort(dist_mat, dim=1)[:, 1:31]
            self.samplers.sample(indexes, sim_mat, targets)
            self.trainer.data_loader = self.trainer.build_active_sample_dataloader(self.samplers.triplet_set, is_train=True)
            self.trainer._data_loader_iter = iter(self.trainer.data_loader)
            
    def get_feature(self):
        num_workers = self.trainer.cfg.DATALOADER.NUM_WORKERS
        batch_size = self.trainer.cfg.TEST.IMS_PER_BATCH
        data_sampler = InferenceSampler(len(self.trainer.data_loader.dataset))
        batch_sampler = torch.utils.data.BatchSampler(data_sampler, batch_size, False)
        dataloader = torch.utils.data.DataLoader(
            self.trainer.data_loader.dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=fast_batch_collator,
        )

        features = []
        targets = []
        model = self.trainer.model
        with inference_context(model), torch.no_grad():
            for idx, inputs in enumerate(dataloader):
                outputs = model(inputs)
                features.append(outputs[0]) # ouputs->Tuple, len=3, 0->pred_feats, 1->targets, 2->camids
                targets.append(outputs[1])
            features = torch.cat(features)
            targets = torch.cat(targets)
        return features, targets
    
    def euclidean_dist(self, x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist
