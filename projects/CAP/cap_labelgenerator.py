import datetime
import logging
import collections

from fastreid.utils import comm
from fastreid.engine.hooks import LabelGeneratorHook

logger = logging.getLogger('fastreid.' + __name__)

class CAPLabelGeneratorHook(LabelGeneratorHook):
    def before_epoch(self):
        if self.trainer.epoch % self._cfg.PSEUDO.CLUSTER_EPOCH == 0 \
                or self.trainer.epoch == self.trainer.start_epoch:
            self._step_timer.reset()

            # generate pseudo labels and centers
            all_labels, all_centers, all_features, all_camids = self.update_labels()
            # update train loader
            self.update_train_loader(all_labels)

            # reset optimizer
            if self._cfg.PSEUDO.RESET_OPT:
                self._logger.info(f"Reset optimizer")
                self.trainer.optimizer.state = collections.defaultdict(dict)

            # update memory labels
            self.trainer.memory._update_centers_and_labels(all_features, all_labels, all_camids)
            self.trainer.memory._update_epoch(self.trainer.epoch)

            # update classifier centers
            try:
                self.update_classifier_centers(all_centers)
            except:
                logger.warning('Error when updating classifier centers.')

            comm.synchronize()

            sec = self._step_timer.seconds()
            self._logger.info(f"Finished updating pseudo label in {str(datetime.timedelta(seconds=int(sec)))}")
