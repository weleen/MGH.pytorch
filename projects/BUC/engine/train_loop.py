# encoding: utf-8
"""
credit:
https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/train_loop.py
"""

import logging
import numpy as np
import time
import weakref
import torch
import fastreid.utils.comm as comm
from fastreid.utils.events import EventStorage
from fastreid.data import samplers
from data.build import fast_batch_collator
from fastreid.evaluation.evaluator import inference_context
from exloss import ExLoss

__all__ = ["HookBase", "TrainerBase", "SimpleTrainer"]


class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.
    Each hook can implement 4 methods. The way they are called is demonstrated
    in the following snippet:
    .. code-block:: python
        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        hook.after_train()
    Notes:
        1. In the hook method, users can access `self.trainer` to access more
           properties about the context (e.g., current iteration).
        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.
           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.
    Attributes:
        trainer: A weak reference to the trainer object. Set by the trainer when the hook is
            registered.
    """

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass


class TrainerBase:
    """
    Base class for iterative trainer with hooks.
    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.
    Attributes:
        iter(int): the current iteration.
        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.
        max_iter(int): The iteration to end training.
        storage(EventStorage): An EventStorage that's opened during the course of training.
    """

    def __init__(self):
        self._hooks = []

    def register_hooks(self, hooks):
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.
        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
            finally:
                self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        for h in self._hooks:
            h.after_train()

    def before_step(self):     
        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()
        # this guarantees, that in each hook's after_step, storage.iter == trainer.iter
        self.storage.step()

    def run_step(self):
        raise NotImplementedError


class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization.
    It assumes that every step, you:
    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.
    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, model, data_loader, train_set, optimizer):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of heads.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()

        self.model = model
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        self.optimizer = optimizer

        self.iters_per_epoch = len(train_set) // self.cfg.SOLVER.IMS_PER_BATCH # cfg.SOLVER.IMS_PER_BATCH
        self.train_set = train_set
        self.loops = 0
        num_train_ids = len(np.unique(np.array(self.train_set.cids)))
        self.criterion = ExLoss(num_train_ids, num_features=2048, t=10).cuda()
        self.nums_to_merge = int(num_train_ids * 0.05) # merge percent

    def after_step(self):
        for h in self._hooks:
            h.after_step()
        # this guarantees, that in each hook's after_step, storage.iter == trainer.iter 
        self.storage.step()
        
        epochs = 20 if self.loops == 0 else 2
        loop = self.iters_per_epoch * epochs
        
        # clustering
        if self.iter > 0 and self.iter % loop == 0:
            if self.loops == 19: return
            new_cids = self.get_new_train_data(self.train_set.cids, self.nums_to_merge, size_penalty=0.003) # size_penalty
            self.train_set.cids = new_cids
            self.loops += 1

    def get_new_train_data(self, labels, nums_to_merge, size_penalty):
        u_feas, label_to_images = self.generate_average_feature(labels)
        
        dists = self.calculate_distance(u_feas)
        
        idx1, idx2 = self.select_merge_data(u_feas, labels, label_to_images, size_penalty, dists)
        
        labels = self.generate_new_train_data(idx1, idx2, labels, nums_to_merge)
        
        num_train_ids = len(np.unique(np.array(labels)))

        # change the criterion classifer
        self.criterion = ExLoss(num_train_ids, num_features=2048, t=10).cuda()

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
        num_workers = self.cfg.DATALOADER.NUM_WORKERS
        batch_size = self.cfg.TEST.IMS_PER_BATCH
        data_sampler = samplers.InferenceSampler(len(self.train_set))
        batch_sampler = torch.utils.data.BatchSampler(data_sampler, batch_size, False)
        dataloader = torch.utils.data.DataLoader(
            self.train_set,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=fast_batch_collator,
        )

        features = []
        model = self.model
        with inference_context(model), torch.no_grad():
            for idx, inputs in enumerate(dataloader):
                outputs = model(inputs)
                features.append(outputs[0]) # ouputs->Tuple, len=3, 0->pred_feats, 1->targets, 2->camids
            features = torch.cat(features)

        return features

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        cids = data['cid'].cuda()
        data_time = time.perf_counter() - start
        """
        If your want to do something with the heads, you can wrap the model.
        """
        outputs = self.model(data)

        # loss_dict = self.model.module.losses(outputs)
        loss_dict = {'loss_excls': self.criterion(outputs[1], cids)[0]} # ouputs->Tuple, len=3, 0->logits, 1->feats, 2->targets
        losses = sum(loss for loss in loss_dict.values())
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        """
        If you need accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        self.optimizer.step()

    def _detect_anomaly(self, losses, loss_dict):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}!\nloss_dict = {}".format(
                    self.iter, loss_dict
                )
            )

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        # if comm.is_main_process():
        if "data_time" in all_metrics_dict[0]:
            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            self.storage.put_scalar("data_time", data_time)

        # average the rest metrics
        metrics_dict = {
            k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())

        self.storage.put_scalar("total_loss", total_losses_reduced)
        if len(metrics_dict) > 1:
            self.storage.put_scalars(**metrics_dict)

