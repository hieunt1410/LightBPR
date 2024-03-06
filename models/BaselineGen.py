import torch
from torch import nn


class BPRmodel(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.num_items = conf['num_item']
        self.emb_dim = conf['dim']
        self.init_emb()

    def init_emb(self):
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.emb_dim))
        nn.init.xavier_normal_(self.items_feature)

    def propagate(self):
        return self.items_feature

    def forward(self, batch):
        item_feature = self.propagate()
        idx, sample_pair = batch
        i_pos, i_neg = sample_pair[:, 0], sample_pair[:, 1]

        item_feat = item_feature[idx]
        pos_feat = item_feature[i_pos]
        neg_feat = item_feature[i_neg]

        pos_score = item_feat @ pos_feat.T
        neg_score = item_feat @ neg_feat.T

        loss = - torch.log(torch.sigmoid(pos_score - neg_score))
        loss = torch.mean(loss)
        return loss


class ItemKNN(nn.Module):
    # return IUI matrix
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

