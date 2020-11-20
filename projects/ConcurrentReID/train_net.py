'''
Author: WuYiming
Date: 2020-10-28 00:11:23
LastEditTime: 2020-11-20 01:03:48
LastEditors: Please set LastEditors
Description: ConcurrentReID
FilePath: /fast-reid/projects/ConcurrentReID/train_net.py
'''
# encoding: utf-8

import sys
import time
import numpy as np
from numpy.core.shape_base import block
import torch
from torch.cuda import amp

import copy

sys.path.append('.')

from fastreid.config import cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fvcore.common.checkpoint import Checkpointer
from fastreid.utils import comm

from concurrentreid import *


class ConcurrentTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It now calls :func:`fastreid.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_reid_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name, val=False):
        """
        Returns:
            iterable
        It now calls :func:`fastreid.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_reid_test_loader(cfg, dataset_name, val)

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
        data_time = time.perf_counter() - start

        """
        If your want to do something with the heads, you can wrap the model.
        """

        with amp.autocast(enabled=self.amp_enabled):
            # manipulate inputs
            data = manipulate_data(data, times=self.cfg.INPUT.MUTUAL.TIMES)
            outs = self.model(data)
            # Compute loss
            if hasattr(self.model, 'module'):
                loss_dict = self.model.module.losses(outs)
            else:
                loss_dict = self.model.losses(outs)

            losses = sum(loss_dict.values())

        with torch.cuda.stream(torch.cuda.Stream()):
            metrics_dict = loss_dict
            metrics_dict["data_time"] = data_time
            self._write_metrics(metrics_dict)
            self._detect_anomaly(losses, loss_dict)

        """
        If you need accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()

        if self.amp_enabled:
            self.scaler.scale(losses).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses.backward()
            """
            If you need gradient clipping/scaling or other processing, you can
            wrap the optimizer with your custom `step()` method.
            """
            self.optimizer.step()

def manipulate_data(data: dict, times:int, concurrent=False) -> dict:
    """
    flatten input images
    """
    new_data = copy.deepcopy(data)
    # unpack all inputs
    images = data['images']
    targets = data['targets']
    camids = data['camids']
    img_paths = data['img_paths']
    index = data['index']
    assert images.size(1) == times
    if concurrent:
        block_size = int(times ** 0.5)
        b, _, c, h, w = images.size()
        # reshape input data
        targets = targets.view(-1, 1).repeat(1, times).view(-1)
        camids = camids.view(-1, 1).repeat(1, times).view(-1)
        img_paths = np.array(img_paths).reshape(-1, 1).repeat(times, axis=1).reshape(-1).tolist()
        index = index.view(-1, 1).repeat(1, times).view(-1)
        images = images.view(-1, *images.size()[2:])
        # permutate data
        seed = comm.shared_random_seed()
        np.random.seed(seed)
        permute_index = np.random.permutation(range(images.size(0))).tolist()
        targets = targets[permute_index]
        camids = camids[permute_index]
        img_paths = [img_paths[i] for i in permute_index]
        index = index[permute_index]
        images = images[permute_index].view(b, times, *images.size()[1:])
        # reshape
        images = images.transpose(1, 2).reshape(b, c, block_size, block_size, h, w).transpose(3, 4).reshape(b, c, block_size * h, block_size * w).contiguous()
    else:
        b, _, c, h, w = images.size()
        images = images.view(-1, c, h, w)
        targets = targets.view(-1, 1).repeat(1, times).view(-1)
        camids = camids.view(-1, 1).repeat(1, times).view(-1)
        img_paths = np.array(img_paths).reshape(-1, 1).repeat(times, axis=1).reshape(-1).tolist()
        index = index.view(-1, 1).repeat(1, times).view(-1)
        
    # regroup images
    new_data['images'] = images
    new_data['targets'] = targets
    new_data['camids'] = camids
    new_data['img_paths'] = img_paths
    new_data['index'] = index
    return new_data
    
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = ConcurrentTrainer.build_model(cfg)
        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = ConcurrentTrainer.test(cfg, model)
        return res

    trainer = ConcurrentTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
