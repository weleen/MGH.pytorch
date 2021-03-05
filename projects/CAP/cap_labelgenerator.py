import datetime
import logging
import collections
import copy

from fastreid.utils import comm
from fastreid.engine.hooks import LabelGeneratorHook
from fastreid.data import build_reid_train_loader

logger = logging.getLogger('fastreid.' + __name__)

class CAPLabelGeneratorHook(LabelGeneratorHook):
    def before_epoch(self):
        if self.trainer.epoch % self._cfg.PSEUDO.CLUSTER_EPOCH == 0 \
                or self.trainer.epoch == self.trainer.start_epoch:
            self._step_timer.reset()

            # generate pseudo labels and centers
            all_labels, all_centers, all_features, all_camids, all_dist_mat = self.update_labels()
            # update train loader
            self.update_train_loader(all_labels)

            if self._cfg.CAP.INSTANCE_LOSS:
                sup_commdataset = self.trainer.data_loader.dataset
                sup_datasets = sup_commdataset.datasets
                pid_labels = list()
                pid_lbls = all_labels[0]
                count = 0
                for pid in pid_lbls:
                    if pid != -1:
                        pid_labels.append(-1)
                    else:
                        pid_labels.append(count)
                        count += 1
                # dataloader for un-clustered samples
                self.trainer.un_data_loader = build_reid_train_loader(self._cfg,
                                                            datasets=copy.deepcopy(sup_datasets),  # copy the sup_datasets
                                                            pseudo_labels=[pid_labels],
                                                            is_train=True,
                                                            relabel=False)
                self.trainer.un_data_loader_iter = iter(self.trainer.un_data_loader)

            # reset optimizer
            if self._cfg.PSEUDO.RESET_OPT:
                self._logger.info(f"Reset optimizer")
                self.trainer.optimizer.state = collections.defaultdict(dict)

            # update memory labels
            self.trainer.memory._update_centers_and_labels(all_features, all_labels, all_camids, all_dist_mat)
            self.trainer.memory._update_epoch(self.trainer.epoch)

            # update classifier centers
            if self._cfg.PSEUDO.WITH_CLASSIFIER:
                self.update_classifier_centers(all_centers)

            comm.synchronize()

            sec = self._step_timer.seconds()
            self._logger.info(f"Finished updating pseudo label in {str(datetime.timedelta(seconds=int(sec)))}")
