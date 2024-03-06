import torch
from torch import nn
from torch.nn.functional import softmax

from models.ZeRe import to_tensor, laplace_transform


class ZeGe(nn.Module):
    def __init__(self, item_feat, conf, graph=None):
        super().__init__()
        # using pre-trained weight
        self.conf = conf
        self.device = conf['device']
        self.item_feature = item_feat
        self.num_item = self.item_feature.shape[0]
        self.iui_graph = graph
        self.ii_propagate_graph = self.get_propagate_graph()

    def get_propagate_graph(self):
        ii_graph = laplace_transform(self.iui_graph)
        ii_graph = to_tensor(ii_graph).to(self.device)
        return ii_graph

    def init_embed(self):
        self.item_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.dim))
        nn.init.xavier_normal_(self.item_feature)

    def propagate(self):
        # item_feats = [self.item_feature]
        # for i in range(self.num_layer):
        #     item_feat = self.ii_propagate_graph @ item_feats[-1]
        #     item_feats.append(item_feat)
        #
        # item_feats = torch.stack(item_feats, dim=0)
        # item_feat = torch.sum(item_feats, dim=0) / (self.num_layer + 1)
        # return item_feat
        return (self.ii_propagate_graph @ self.item_feature + self.item_feature) / 2

    def forward(self, batch):
        # batch [id, [pos, neg]]
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

    def get_item_feat(self):
        return self.item_feature

    @torch.no_grad()
    def gen(self, k=None):
        if k is None:
            k = self.num_item
        item_correl = self.item_feature @ self.item_feature.T
        item_correl = softmax(item_correl.fill_diagonal_(-1e8), dim=1)
        top_k_prob, top_k_bundle = torch.topk(item_correl, k=k - 1, dim=1)  # not include self item

        indices = torch.arange(0, self.num_item).view(-1, 1)
        id_prob = torch.zeros((self.num_item, 1))

        bundle = torch.cat([indices, top_k_bundle.to('cpu')], dim=1)
        prob = torch.cat([id_prob, top_k_prob.to('cpu')], dim=1)
        return bundle, prob

    def semantic_filter(self):
        pass

    def rm_dup_bundle(self, bundles, topk):
        # remember that each bundle need 1 item example index 720 : [1,2,3,4]
        # -> bundle is [720, 1, 2, 3, 4]
        bundles = bundles[:, :topk]
        bundles, _ = bundles.sort(dim=1)
        bundles = bundles.unique(dim=0)
        return bundles
