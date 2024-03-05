#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt

    return graph


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))

    return graph


def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1 - dropout_ratio])
    values = mask * values
    return values


class ZeRe(nn.Module):
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

        assert isinstance(raw_graph, list)
        self.ui_graph, self.bi_graph = raw_graph

        # generate the graph without any dropouts for testing
        self.get_item_level_graph_ori()
        self.get_bundle_agg_graph_ori()
        self.get_user_agg_graph_ori()

        # generate the graph with the configured dropouts for training, if aug_type is OP or MD, the following graphs with be identical with the aboves
        self.get_item_level_graph()
        self.get_bundle_agg_graph()
        self.get_user_agg_graph()

        self.init_md_dropouts()
        self.num_layers = 1

        self.ibi_graph = (self.bi_graph.T @ self.bi_graph) > 0
        self.iui_graph = (self.ui_graph.T @ self.ui_graph) > 0
        self.ibi_mask = to_tensor(self.ibi_graph).to_dense().to(self.device)
        self.iui_mask = to_tensor(self.iui_graph).to_dense().to(self.device)
        self.get_item_item_propagate_graph()

        self.q = nn.Linear(self.embedding_size, self.embedding_size)
        self.k = nn.Linear(self.embedding_size, self.embedding_size)
        self.v = nn.Linear(self.embedding_size, self.embedding_size)

    def calculate_1head_attention(self, mask):
        items = self.items_feature
        norm = self.embedding_size ** 0.5
        wq = self.q(items)
        wk = self.k(items)
        wv = self.v(items)
        res = F.softmax((wq @ wk.T * mask / norm), dim=1) @ wv
        return res

    def cal_in_bundle_attn(self):
        bi_mat = to_tensor(self.bi_graph)
        norm = (bi_mat.sum(dim=1) + 1e-8).reshape(-1, 1)
        bi_mat = bi_mat.unsqueeze(dim=1).transpose(dim0=1, dim1=2)
        item_in_bundle_feature = self.items_feature * bi_mat

        iq = self.q(item_in_bundle_feature)
        ik = self.k(item_in_bundle_feature)
        iv = self.v(item_in_bundle_feature)

        attn_i = F.softmax((iq @ ik.transpose(dim0=1, dim1=2) / self.embedding_size ** -0.5), dim=2) @ iv
        item_out = torch.sum(attn_i, dim=-1) / norm
        return item_out

    def init_md_dropouts(self):
        self.item_level_dropout = nn.Dropout(self.conf["item_level_ratio"], True)
        self.bundle_level_dropout = nn.Dropout(self.conf["bundle_level_ratio"], True)
        self.bundle_agg_dropout = nn.Dropout(self.conf["bundle_agg_ratio"], True)

    def init_emb(self):
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)

    def get_item_item_propagate_graph(self):
        ibi_graph = self.ibi_graph
        self.ibi_propagate_graph = to_tensor(laplace_transform(ibi_graph))

        iui_graph = self.iui_graph
        self.iui_propagate_graph = to_tensor(laplace_transform(iui_graph))

    def get_item_level_graph(self):
        ui_graph = self.ui_graph
        bi_graph = self.bi_graph
        device = self.device
        modification_ratio = self.conf["item_level_ratio"]

        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph],
                                    [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
        bi_propagate_graph = sp.bmat([[sp.csr_matrix((bi_graph.shape[0], bi_graph.shape[0])), bi_graph],
                                      [bi_graph.T, sp.csr_matrix((bi_graph.shape[1], bi_graph.shape[1]))]])
        self.bi_propagate_graph_ori = to_tensor(laplace_transform(bi_propagate_graph)).to(device)

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = item_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                item_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

                graph2 = bi_propagate_graph.tocoo()
                values2 = np_edge_dropout(graph2.data, modification_ratio)
                bi_propagate_graph = sp.coo_matrix((values2, (graph2.row, graph2.col)), shape=graph2.shape).tocsr()

        self.item_level_graph = to_tensor(laplace_transform(item_level_graph)).to(device)
        self.bi_propagate_graph = to_tensor(laplace_transform(bi_propagate_graph)).to(device)

    def get_item_level_graph_ori(self):
        ui_graph = self.ui_graph
        device = self.device
        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph],
                                    [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
        self.item_level_graph_ori = to_tensor(laplace_transform(item_level_graph)).to(device)

    def get_bundle_agg_graph(self):
        bi_graph = self.bi_graph
        device = self.device

        if self.conf["aug_type"] == "ED":
            modification_ratio = self.conf["bundle_agg_ratio"]
            graph = self.bi_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            bi_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1 / bundle_size.A.ravel()) @ bi_graph
        self.bundle_agg_graph = to_tensor(bi_graph).to(device)

    def get_user_agg_graph(self):
        ui_graph = self.ui_graph
        device = self.device

        if self.conf["aug_type"] == "ED":
            modification_ratio = self.conf["bundle_agg_ratio"]
            graph = self.ui_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            ui_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        user_size = ui_graph.sum(axis=1) + 1e-8
        ui_graph = sp.diags(1 / user_size.A.ravel()) @ ui_graph
        self.user_agg_graph = to_tensor(ui_graph).to(device)

    def get_bundle_agg_graph_ori(self):
        bi_graph = self.bi_graph
        device = self.device

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1 / bundle_size.A.ravel()) @ bi_graph
        self.bundle_agg_graph_ori = to_tensor(bi_graph).to(device)

    def get_user_agg_graph_ori(self):
        ui_graph = self.ui_graph
        user_size = ui_graph.sum(axis=1) + 1e-8
        ui_graph = sp.diags(1 / user_size.A.ravel()) @ ui_graph
        self.user_agg_graph_ori = to_tensor(ui_graph).to(self.device)

    def one_propagate(self, graph, A_feature, B_feature, mess_dropout, test, coefs=None):
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]

        for i in range(self.num_layers):
            features = torch.spmm(graph, features)
            if self.conf["aug_type"] == "MD" and not test:  # !!! important
                features = mess_dropout(features)

            features = features / (i + 2)
            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1)
        if coefs is not None:
            all_features = all_features * coefs
        all_features = torch.sum(all_features, dim=1).squeeze(1)

        A_feature, B_feature = torch.split(all_features, [A_feature.shape[0], B_feature.shape[0]], 0)

        return A_feature, B_feature

    def get_IL_bundle_rep(self, IL_items_feature, test):
        if test:
            IL_bundles_feature = torch.matmul(self.bundle_agg_graph_ori, IL_items_feature)
        else:
            IL_bundles_feature = torch.matmul(self.bundle_agg_graph, IL_items_feature)

        # simple embedding dropout on bundle embeddings
        if self.conf["bundle_agg_ratio"] != 0 and self.conf["aug_type"] == "MD" and not test:
            IL_bundles_feature = self.bundle_agg_dropout(IL_bundles_feature)

        return IL_bundles_feature

    def get_IL_user_rep(self, IL_items_feature, test):
        if test:
            IL_users_feature = torch.matmul(self.user_agg_graph_ori, IL_items_feature)
        else:
            IL_users_feature = torch.matmul(self.user_agg_graph, IL_items_feature)

        # simple embedding dropout on bundle embeddings
        if self.conf["bundle_agg_ratio"] != 0 and self.conf["aug_type"] == "MD" and not test:
            IL_users_feature = self.bundle_agg_dropout(IL_users_feature)

        return IL_users_feature

    def propagate(self, test=False):
        #  ======== UI =================
        if test:
            IL_users_feature, IL_items_feature = self.one_propagate(self.item_level_graph_ori, self.users_feature,
                                                                    self.items_feature, self.item_level_dropout, test,
                                                                    None)
        else:
            IL_users_feature, IL_items_feature = self.one_propagate(self.item_level_graph, self.users_feature,
                                                                    self.items_feature, self.item_level_dropout, test,
                                                                    None)

        # item_Attn = self.calculate_1head_attention(self.ibi_mask)
        # return IL_users_feature, (IL_items_feature + item_Attn) / 2
        return IL_users_feature, IL_items_feature

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

    # @torch.no_grad()
    def save_check_point(self):
        print('saving feature for bundle generation task!')
        user_feat, item_feat = self.propagate(test=False)
        torch.save(item_feat, os.path.join(self.conf['data_path'], self.conf['dataset'], 'pretrain_feature.pt'))
        torch.save(self.items_feature, os.path.join(self.conf['data_path'], self.conf['dataset'], 'self_trained_feature.pt'))