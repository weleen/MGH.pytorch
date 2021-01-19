# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import sys
import logging
from collections import OrderedDict

sys.path.append('.')
from fastreid.config import cfg
from fastreid.engine import (DefaultTrainer, default_argument_parser,
                             default_setup, launch)
from fastreid.evaluation import DatasetEvaluator, print_csv_format
from fvcore.common.checkpoint import Checkpointer

from transformer.trans_baseline import Trans_Baseline
from utils import inference_on_dataset
from fastreid.utils import comm
from fastreid.engine import hooks
import torch


class TRANSTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(TRANSTrainer, self).__init__(cfg)
    
    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        logger = logging.getLogger(__name__)
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN
        cfg.DATASETS.NAMES = tuple([cfg.TEST.PRECISE_BN.DATASET])  # set dataset name for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
        ]

        if cfg.PSEUDO.ENABLED:
            assert len(cfg.PSEUDO.UNSUP) > 0, "there are no dataset for unsupervised learning"
            ret.append(
                hooks.LabelGeneratorHook(
                    self.cfg,
                    self.model
                )
            )

        if cfg.SOLVER.SWA.ENABLED:
            ret.append(
                hooks.SWA(
                    cfg.SOLVER.MAX_ITER,
                    cfg.SOLVER.SWA.PERIOD,
                    cfg.SOLVER.SWA.LR_FACTOR,
                    cfg.SOLVER.SWA.ETA_MIN_LR,
                    cfg.SOLVER.SWA.LR_SCHED,
                )
            )

        if cfg.TEST.PRECISE_BN.ENABLED and hooks.get_bn_modules(self.model):
            logger.info("Prepare precise BN dataset")
            ret.append(hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            ))

        if cfg.MODEL.OPEN_LAYERS != [''] and cfg.SOLVER.FREEZE_ITERS > 0:
            open_layers = ",".join(cfg.MODEL.OPEN_LAYERS)
            logger.info(f'Open "{open_layers}" training for {cfg.SOLVER.FREEZE_ITERS:d} iters')
            ret.append(hooks.FreezeLayer(
                self.model,
                cfg.MODEL.OPEN_LAYERS,
                cfg.SOLVER.FREEZE_ITERS,
            ))
        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results(val=False):
            self._last_eval_results = self.test_transformer(self.cfg, self.model, val=val)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results, do_val=self.cfg.TEST.DO_VAL, metric_names=self.cfg.TEST.METRIC_NAMES))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), cfg.SOLVER.LOG_PERIOD))

        return ret

    def test_transformer(cls, cfg, model, evaluators=None, val=False):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TESTS`.
            val (bool): if True, perform validation on val split. Otherwise, test on query and gallery split.
        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger('fastreid.' + __name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]

        if evaluators is not None:
            assert len(cfg.DATASETS.TESTS) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TESTS), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TESTS):
            logger.info("Prepare testing set")
            data_loader, num_query = cls.build_test_loader(cfg, dataset_name, val)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, num_query)
                except NotImplementedError:
                    logger.warning(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i

        if comm.is_main_process():
            assert isinstance(
                results, dict
            ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                results
            )
            print_csv_format(results)

        if len(results) == 1: results = list(results.values())[0]

        return results


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
        model = TRANSTrainer.build_model(cfg)
        Checkpointer(model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS)  # load trained model
        res = TRANSTrainer.test(cfg, model)
        return res

    trainer = TRANSTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    # trainer.test(trainer.cfg, trainer.model)
    # trainer.test_transformer(trainer.cfg, trainer.model)
    # torch.cuda.empty_cache()
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
