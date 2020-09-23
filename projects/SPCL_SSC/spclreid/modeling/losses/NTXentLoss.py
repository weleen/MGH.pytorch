import torch
from fastreid.utils import comm


class NTXentLoss(torch.nn.Module):
    def __init__(self, cfg):
        super(NTXentLoss, self).__init__()
        self.batch_size = cfg.SOLVER.IMS_PER_BATCH
        self.temperature = cfg.UNSUPERVISED.LOSS.TEMP
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.sim_f = torch.nn.CosineSimilarity(dim=2)

    def __call__(self, z_i, z_j, pseudo_id):
        # pseudo_id should be all different in the batch_size
        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)
        pseudo_id = torch.cat((pseudo_id, pseudo_id), dim=0)
        if comm.get_world_size() > 1:
            z = comm.concat_all_gather(z)
            pseudo_id = comm.concat_all_gather(pseudo_id)

        mask = pseudo_id.unsqueeze(0) == pseudo_id.unsqueeze(1)
        neg_mask = ~mask
        pos_mask = mask.fill_diagonal_(0)
        sim = self.sim_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = sim[pos_mask].reshape(N, 1)
        negative_samples = sim[neg_mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return {"NTXentLoss": loss}
