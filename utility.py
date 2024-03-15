#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader


def graph_to_pairs(graph):
    graph = graph.tocoo()
    indices = np.array([graph.row, graph.col]).T
    return indices


class BundleTrainDataset(Dataset):
    def __init__(self, conf, u_b_pairs, u_b_graph, num_bundles, u_b_for_neg_sample, b_b_for_neg_sample, neg_sample=1):
        self.conf = conf
        self.u_b_pairs = u_b_pairs
        self.u_b_graph = u_b_graph
        self.num_bundles = num_bundles
        self.neg_sample = neg_sample

        self.u_b_for_neg_sample = u_b_for_neg_sample
        self.b_b_for_neg_sample = b_b_for_neg_sample

    def __getitem__(self, index):
        conf = self.conf
        user_b, pos_bundle = self.u_b_pairs[index]
        all_bundles = [pos_bundle]

        while True:
            i = np.random.randint(self.num_bundles)
            if self.u_b_graph[user_b, i] == 0 and i not in all_bundles:
                all_bundles.append(i)
                if len(all_bundles) == self.neg_sample + 1:
                    break

        return torch.LongTensor([user_b]), torch.LongTensor(all_bundles)

    def __len__(self):
        return len(self.u_b_pairs)


class BunGenDataset(Dataset):
    def __init__(self, conf, ii_pairs, i_i_graph, num_items):
        self.conf = conf
        self.ii_pairs = ii_pairs
        self.i_i_graph = i_i_graph
        self.num_items = num_items

    def __getitem__(self, item):
        idx, pos_i = self.ii_pairs[item]

        while True:
            neg_i = np.random.randint(self.num_items)
            if self.i_i_graph[idx, neg_i] == 0:
                break

        return torch.LongTensor([idx]), torch.LongTensor([pos_i, neg_i])

    def __len__(self):
        return len(self.ii_pairs)


class ItemTrainDataset(Dataset):
    def __init__(self, conf, u_i_pairs, u_i_graph, num_items, neg_sample=1):
        self.conf = conf
        self.u_i_pairs = u_i_pairs
        self.u_i_graph = u_i_graph
        self.num_items = num_items
        self.neg_sample = neg_sample

    def __getitem__(self, index):
        conf = self.conf
        user_b, pos_item = self.u_i_pairs[index]
        all_items = [pos_item]

        while True:
            i = np.random.randint(self.num_items)
            if self.u_i_graph[user_b, i] == 0 and not i in all_items:
                all_items.append(i)
                if len(all_items) == self.neg_sample + 1:
                    break

        return torch.LongTensor([user_b]), torch.LongTensor(all_items)

    def __len__(self):
        return len(self.u_i_pairs)


class BundleTestDataset(Dataset):
    def __init__(self, u_b_pairs, u_b_graph, u_b_graph_train, num_users, num_bundles):
        self.u_b_pairs = u_b_pairs
        self.u_b_graph = u_b_graph
        self.train_mask_u_b = u_b_graph_train

        self.num_users = num_users
        self.num_bundles = num_bundles

        self.users = torch.arange(num_users, dtype=torch.long).unsqueeze(dim=1)
        self.bundles = torch.arange(num_bundles, dtype=torch.long)

    def __getitem__(self, index):
        u_b_grd = torch.from_numpy(self.u_b_graph[index].toarray()).squeeze()
        u_b_mask = torch.from_numpy(self.train_mask_u_b[index].toarray()).squeeze()

        return index, u_b_grd, u_b_mask

    def __len__(self):
        return self.u_b_graph.shape[0]


class Datasets():
    def __init__(self, conf):
        self.path = conf['data_path']
        self.name = conf['dataset']
        batch_size_train = conf['batch_size_train']
        batch_size_test = conf['batch_size_test']
        self.sep = conf['sep']
        self.file_type = conf['file_type']

        self.num_users, self.num_bundles, self.num_items = self.get_data_size()

        b_i_graph = self.get_bi()
        u_i_pairs, u_i_graph = self.get_ui()

        u_b_pairs_val, u_b_graph_val = self.get_ub("valid")
        u_b_pairs_test, u_b_graph_test = self.get_ub("test")
        u_b_pairs_train, u_b_graph_train = self.get_ub("train")
        iui_graph = (u_i_graph.T @ u_i_graph - sp.eye(self.num_items, self.num_items)) > 0

        self.iui_graph = iui_graph
        self.bi_graph = b_i_graph

        self.bundle_val_data = BundleTestDataset(u_b_pairs_val, u_b_graph_val, u_b_graph_train, self.num_users,
                                                 self.num_bundles)
        self.bundle_test_data = BundleTestDataset(u_b_pairs_test, u_b_graph_test, u_b_graph_train, self.num_users,
                                                  self.num_bundles)
        self.item_train_data = ItemTrainDataset(conf, u_i_pairs, u_i_graph, self.num_items, neg_sample=conf['neg_num'])
        self.item_item_train_data = BunGenDataset(conf, graph_to_pairs(iui_graph), iui_graph, self.num_items)
        self.graphs = [u_i_graph, b_i_graph]
        self.item_train_loader = DataLoader(self.item_train_data, batch_size=batch_size_train, shuffle=True,
                                            num_workers=4, drop_last=False)
        self.val_loader = DataLoader(self.bundle_val_data, batch_size=batch_size_test, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(self.bundle_test_data, batch_size=batch_size_test, shuffle=False, num_workers=4)

        self.item_item_loader = DataLoader(self.item_item_train_data, batch_size=batch_size_train, shuffle=False,
                                           num_workers=4)

    def get_data_size(self):
        name = self.name
        if "_" in name:
            name = name.split("_")[0]
        with open(os.path.join(self.path, self.name, '{}_data_size.txt'.format(name)), 'r') as f:
            return [int(s) for s in f.readline().split('\t')][:3]

    def get_bi(self):
        with open(os.path.join(self.path, self.name, 'bundle_item' + self.file_type), 'r') as f:
            b_i_pairs = list(
                map(lambda s: tuple(int(i) for i in s[:-1].split(self.sep))[:2], f.readlines()))  # don't get timestamp

        indices = np.array(b_i_pairs, dtype=np.int32)
        values = np.ones(len(b_i_pairs), dtype=np.float32)
        b_i_graph = sp.coo_matrix(
            (values, (indices[:, 0], indices[:, 1])), shape=(self.num_bundles, self.num_items)).tocsr()

        return b_i_graph

    def get_ui(self):
        with open(os.path.join(self.path, self.name, 'user_item' + self.file_type), 'r') as f:
            u_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split(self.sep))[:2], f.readlines()))

        indices = np.array(u_i_pairs, dtype=np.int32)
        values = np.ones(len(u_i_pairs), dtype=np.float32)
        u_i_graph = sp.coo_matrix(
            (values, (indices[:, 0], indices[:, 1])), shape=(self.num_users, self.num_items)).tocsr()

        return u_i_pairs, u_i_graph

    def get_ub(self, task):
        with open(os.path.join(self.path, self.name, 'user_bundle_{}'.format(task) + self.file_type), 'r') as f:
            u_b_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split(self.sep))[:2], f.readlines()))

        indices = np.array(u_b_pairs, dtype=np.int32)
        values = np.ones(len(u_b_pairs), dtype=np.float32)
        u_b_graph = sp.coo_matrix(
            (values, (indices[:, 0], indices[:, 1])), shape=(self.num_users, self.num_bundles)).tocsr()

        return u_b_pairs, u_b_graph


if __name__ == '__main__':
    dataset = Datasets(conf={
        'dataset': 'clothing',
        'data_path': './datasets',
        'batch_size_train': 128,
        'batch_size_test': 128,
        'topk': [1, 2, 3, 4],
        'neg_num': 1,
        'aug_type': "ED",
        'ed_interval': 1,
        'embedding_sizes': [64],
        'num_layerss': [1],
        'item_level_ratios': [0.0],
        'bundle_level_ratios': [0.0],
        'bundle_agg_ratios': [0.0],
        'lrs': [1.0e-3],
        'l2_regs': [1.0e-4],
        'epochs': 50,
        'test_interval': 1,
        'sep': "\t",
        'file_type': ".txt",
        'topk_valid': 3,
    })

    print(len(dataset.item_item_loader))
