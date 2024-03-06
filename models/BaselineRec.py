import torch
from torch import nn


def cal_bpr_loss(pred, alpha=0.2):
    # pred: [bs, 1+neg_num]
    if pred.shape[1] > 2:
        negs = pred[:, 1:]
        pos = pred[:, 0].unsqueeze(1).expand_as(negs)
    else:
        negs = pred[:, 1].unsqueeze(1)
        pos = pred[:, 0].unsqueeze(1)

    loss = - torch.log(torch.sigmoid(pos - negs))  # [bs]
    loss = torch.mean(loss)
    return loss


class BPRModel(nn.Module):
    def __init__(self, conf, raw_graph):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device

        self.embedding_size = conf["embedding_size"]
        self.embed_L2_norm = conf["l2_reg"]
        self.num_users = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items = conf["num_items"]
        self.init_emb()

    def init_emb(self):
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)

    def propagate(self, test=False):
        return self.users_feature, self.items_feature

    def cal_loss(self, users_feature, bundles_feature):
        users_feature = users_feature
        bundles_feature = bundles_feature
        pred = torch.sum(users_feature * bundles_feature, 2)
        bpr_loss = cal_bpr_loss(pred)
        return bpr_loss

    def forward(self, batch, ED_drop=False):
        # the edge drop can be performed by every batch or epoch, should be controlled in the train loop
        if ED_drop:
            self.get_item_level_graph()
            self.get_bundle_agg_graph()

        # users: [bs, 1]
        # bundles: [bs, 1+neg_num]
        users, items = batch
        users_feature, items_feature = self.propagate()

        users_embedding = users_feature[users]
        items_embedding = items_feature[items]
        bpr_loss = self.cal_loss(users_embedding, items_embedding)
        return bpr_loss

    def evaluate(self, propagate_result, users):
        users_feature, items_feature = propagate_result
        users_feature = users_feature[users]
        # aggregate items to bundle
        bundles_feature = self.get_IL_bundle_rep(items_feature, test=False)
        scores = torch.mm(users_feature, bundles_feature.t())
        return scores
