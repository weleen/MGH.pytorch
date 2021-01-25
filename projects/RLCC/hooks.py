import os
import datetime
import collections
import torch

from fastreid.utils import comm
from fastreid.engine.hooks import *


class RLCCLabelGeneratorHook(LabelGeneratorHook):
    """ Hook for RLCC.
    """
    def before_epoch(self):
        if self.trainer.epoch % self._cfg.PSEUDO.CLUSTER_EPOCH == 0 \
                or self.trainer.epoch == self.trainer.start_epoch:
            self._step_timer.reset()

            if self._cfg.PSEUDO.RLCC.ENABLED:
                self.save_last_features_labels_centers()

            # get memory features
            self.get_memory_features()

            # generate pseudo labels and centers
            all_labels, all_centers = self.update_labels()

            # update train loader
            self.update_train_loader(all_labels)

            # reset optimizer
            if self._cfg.PSEUDO.RESET_OPT:
                self._logger.info(f"Reset optimizer")
                self.trainer.optimizer.state = collections.defaultdict(dict)

            if hasattr(self.trainer, 'memory'):
                # update memory labels, memory based methods such as SpCL
                self.update_memory_labels(all_labels)
                assert len(all_centers) == 1, 'only support single unsupervised dataset'
                self.trainer.memory._update_center(all_centers[0])
            else:
                # update classifier centers, methods such as SBL
                self.update_classifier_centers(all_centers)

            comm.synchronize()

            if self._cfg.PSEUDO.RLCC.ENABLED and self.trainer.epoch >= self._cfg.PSEUDO.RLCC.START_EPOCH:  # start label refinery
                filename = os.path.join(self._cfg.OUTPUT_DIR, 'clustering/labels_epoch{}.pt'.format(self.trainer.epoch))
                self.trainer.memory.calculate_mapping_matrix(filename)

            sec = self._step_timer.seconds()
            self._logger.info(f"Finished updating pseudo label in {str(datetime.timedelta(seconds=int(sec)))}")

    def save_last_features_labels_centers(self, features=None, labels=None, centers=None):
        saved_dict = {}
        if features is None:
            assert hasattr(self.trainer, 'memory'), "Only support memory-based approaches now."
            features = self.trainer.memory.features
        self.trainer.memory._update_last_feature(features)
        saved_dict.update({'last_features': features})

        if labels is None:
            assert hasattr(self.trainer, 'memory'), "Only support memory-based approaches now."
            labels = self.trainer.memory.labels
        self.trainer.memory._update_last_label(labels)
        saved_dict.update({'last_labels': labels})
        
        if centers is None:
            assert hasattr(self.trainer, 'memory'), "Only support memory-based approaches now."
            if self.trainer.epoch == 0:
                centers = self.trainer.memory.features
            else:
                centers = self.trainer.memory.centers
        self.trainer.memory._update_last_center(centers)
        saved_dict.update({'last_centers': centers})
