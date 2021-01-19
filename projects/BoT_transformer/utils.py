import copy
import datetime
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from fastreid.utils import comm
from fastreid.utils.logger import log_every_n_seconds


def inference_on_dataset(model, data_loader, evaluator):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.
    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.
    Returns:
        The return value of `evaluator.evaluate()`
    """
    logger = logging.getLogger('fastreid.' + __name__)
    logger.info("Start inference on {} images".format(len(data_loader.dataset)))

    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            idx += 1
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_batch = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_batch > 30:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / batch. ETA={}".format(
                        idx + 1, total, seconds_per_batch, str(eta)
                    ),
                    n=30,
                    name='fastreid.' + __name__
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / batch per device)".format(
            total_time_str, total_time / (total - num_warmup)
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / batch per device)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup)
        )
    )
    torch.cuda.empty_cache()
    results = evaluate(evaluator, model)
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


def evaluate(self, model):
    if comm.get_world_size() > 1:
        comm.synchronize()
        features = comm.gather(self.features)
        features = sum(features, [])

        pids = comm.gather(self.pids)
        pids = sum(pids, [])

        camids = comm.gather(self.camids)
        camids = sum(camids, [])

        # fmt: off
        if not comm.is_main_process(): return {}
        # fmt: on
    else:
        features = self.features
        pids = self.pids
        camids = self.camids

    features = torch.cat(features, dim=0)
    # query feature, person ids and camera ids
    query_features = features[:self._num_query]
    query_pids = np.asarray(pids[:self._num_query])
    query_camids = np.asarray(camids[:self._num_query])

    # gallery features, person ids and camera ids
    gallery_features = features[self._num_query:]
    gallery_pids = np.asarray(pids[self._num_query:])
    gallery_camids = np.asarray(camids[self._num_query:])

    self._results = OrderedDict()

    if self.cfg.TEST.METRIC == "cosine":
        query_features = F.normalize(query_features, dim=1)
        gallery_features = F.normalize(gallery_features, dim=1)

    with inference_context(model), torch.no_grad():
        # if hasattr(model, 'module'):
        #     dist = []
        #     for q in query_features:
        #         d, indx = model.module.transformer_inference(q.unsqueeze(0), gallery_features)
        #         dist.append(d)
        #     dist = torch.cat(dist, 0)
        # else:
        #     dist = []
        #     for q in query_features:
        #         d, indx = model.transformer_inference(q.unsqueeze(0), gallery_features)
        #         dist.append(d)
        #     dist = torch.cat(dist, 0)
        dist, indx = model.module.transformer_inference(query_features, gallery_features) 
        dist = dist.cpu().numpy()
        indx = indx.cpu().numpy()

    query_features = query_features.numpy()
    gallery_features = gallery_features.numpy()
    cmc, all_AP, all_INP = evaluate_rank(dist, indx, query_features, gallery_features,
                                            query_pids, gallery_pids, query_camids, gallery_camids,
                                            use_distmat=True)

    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    for r in [1, 5, 10]:
        self._results['Rank-{}'.format(r)] = cmc[r - 1]
    self._results['mAP'] = mAP
    self._results['mINP'] = mINP

    return copy.deepcopy(self._results)


def evaluate_rank(distmat, indx, q_feats, g_feats, q_pids, g_pids, q_camids, g_camids, max_rank=50, use_distmat=True):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    dim = q_feats.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(g_feats)

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    if use_distmat:
        indices = np.argsort(distmat, axis=1)
    else:
        _, indices = index.search(q_feats, k=num_g)

    g_pids_mat = g_pids[indx]
    for i in range(len(g_pids_mat)):
        g_pids_mat[i] = g_pids_mat[i][indices[i]]
    matches = (g_pids_mat == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    return all_cmc, all_AP, all_INP


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.
    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
