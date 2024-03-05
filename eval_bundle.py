import os
from argparse import ArgumentParser

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm


def get_arg():
    argp = ArgumentParser()
    argp.add_argument('-d', '--dataset', type=str, default='clothing')
    args = argp.parse_args()
    return args


def get_data_size(path="datasets/", dataset="clothing"):
    if "_" in dataset:
        dataset = dataset.split("_")[0]
    with open(os.path.join(path, dataset, '{}_data_size.txt'.format(dataset)), 'r') as f:
        return [int(s) for s in f.readline().split('\t')][:3]


def get_bi(path="datasets/", dataset="clothing", file_type=".txt", shape=None):
    with open(os.path.join(path, dataset, 'bundle_item' + file_type), 'r') as f:
        b_i_pairs = list(
            map(lambda s: tuple(int(i) for i in s[:-1].split("\t"))[:2], f.readlines()))  # don't get timestamp

    indice = np.array(b_i_pairs, dtype=np.int32)
    values = np.ones(len(b_i_pairs), dtype=np.float32)
    b_i_graph = sp.coo_matrix(
        (values, (indice[:, 0], indice[:, 1])), shape=shape).tocsr()

    return b_i_graph


def bundle_graph2list(bi_graph):
    bi_graph = bi_graph.tocoo()
    bundle, item = bi_graph.row, bi_graph.col
    bundle_dict = {}

    for i in range(0, len(bundle)):
        if bundle[i] not in bundle_dict.keys():
            bundle_dict[bundle[i]] = [item[i]]
        else:
            bundle_dict[bundle[i]].append(item[i])

    # print(bundle_dict)
    all_bundles = []
    for key in bundle_dict.keys():
        all_bundles.append(bundle_dict[key])

    for i in all_bundles:
        i.sort()
    return all_bundles


def rm_dup_bundle(bundles, size):
    # remember that each bundle need 1 item example index 720 : [1,2,3,4]
    # -> bundle is [720, 1, 2, 3, 4]
    bundles = bundles[:, :size]
    bundles, _ = bundles.sort(dim=1)
    bundles = bundles.unique(dim=0)
    return bundles.tolist()


def recall(pred_list, gd_list):
    tp = 0
    pbar = tqdm(gd_list)
    for i in pbar:
        for j in pred_list:
            if i == j:
                tp += 1
    return tp / len(gd_list)


def precision(pred_list, gd_list):
    tp = 0
    pbar = tqdm(gd_list)
    for i in pbar:
        for j in pred_list:
            if i == j:
                tp += 1
    return tp / len(pred_list)


def metrics(pred_list, gd_list):
    tp = 0
    pbar = tqdm(gd_list)
    for i in pbar:
        for j in pred_list:
            if i == j:
                tp += 1

    re = tp / len(gd_list)
    pre = tp / len(pred_list)
    print('generated_size: %i, true positive: %i, recall %.4f, precision %.4f' % (len(pred_list[0]), tp, re, pre))
    return {
        'precision': pre,
        'recall': re,
    }


if __name__ == '__main__':
    args = get_arg()
    n_u, n_b, n_i = get_data_size(dataset=args.dataset)
    bundle = torch.load(f'datasets/{args.dataset}/bundle.pt')

    # print(bundle[720])

    size5_unique_bundle = rm_dup_bundle(bundle, size=5)
    size4_unique_bundle = rm_dup_bundle(bundle, size=4)
    size3_unique_bundle = rm_dup_bundle(bundle, size=3)
    size2_unique_bundle = rm_dup_bundle(bundle, size=2)
    size6_unique_bundle = rm_dup_bundle(bundle, size=6)

    bi_list = bundle_graph2list(get_bi(dataset=args.dataset, shape=(n_b, n_i)))
    print(f'num of bundles: {len(bi_list)}')
    metrics(size2_unique_bundle, bi_list)
    metrics(size3_unique_bundle, bi_list)
    metrics(size4_unique_bundle, bi_list)
    metrics(size5_unique_bundle, bi_list)
    metrics(size6_unique_bundle, bi_list)
