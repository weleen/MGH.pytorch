from fastreid.engine.train_loop import HookBase
from fastreid.engine.hooks import *
from samplers import build_active_samplers


class DataloaderHook(HookBase):
    def __init__(self, cfg, labeled_set, unlabeled_set):
        super().__init__()
        self.sample_iter = cfg.ACTIVE.TRAIN_CYCLES
        labeled_num = int(len(unlabeled_set) * cfg.ACTIVE.INITIAL_RATE)
        self.samplers = build_active_samplers(cfg, labeled_set, unlabeled_set, labeled_num)

    def before_step(self):
        if self.trainer.iter % self.sample_iter == 0:

            self.samplers.sample()

            self.trainer.data_loader = self.trainer.build_active_sample_dataloader(self.samplers.labeled_set, is_train=True)
            self.trainer._data_loader_iter = iter(self.trainer.data_loader)
