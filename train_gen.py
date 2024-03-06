from argparse import ArgumentParser

import yaml
from torch.optim import Adam
from tqdm import tqdm

from models.ZeGe import *
from utility import *


def load_config(dataset):
    f = open('configs/gen.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configs = yaml.safe_load(f)
    config = configs[dataset]
    config['dataset'] = dataset
    config['device'] = device
    return config


def get_arg():
    argp = ArgumentParser()
    argp.add_argument('-d', '--dataset', type=str, default='clothing')
    argp.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    argp.add_argument('-ep', '--epochs', type=int, default=30)
    argp.add_argument('-bs', '--bundle_size', type=int, default=10)

    args = argp.parse_args()
    return args


def train_gen_bundle(model, optimizer, dataloader, conf, gt_bundle):
    EPOCH = conf['epochs']
    device = conf['device']

    for epoch in range(0, EPOCH + 1):
        model.train(True)
        pbar = tqdm(dataloader, total=len(dataloader))

        for i, batch in enumerate(pbar):
            batch = [x.to(device) for x in batch]
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            pbar.set_description('Epoch: %i, Loss: %.4f' % (epoch, loss))

        model.eval()
        gen_bundle, prob = model.gen(conf['bundle_size'])
        eval_generated_bundle(gen_bundle, gt_bundle)

    model.eval()
    res_bundle, prob = model.gen(conf['bundle_size'])
    torch.save(res_bundle, os.path.join(conf['data_path'], conf['dataset'], 'bundle.pt'))


def metrics(pred_list, gd_list):
    tp = 0
    pbar = tqdm(gd_list)
    for i in pbar:
        for j in pred_list:
            tp += (i == j)

    re = tp / len(gd_list)
    pre = tp / len(pred_list)
    print('generated_size: %i, true positive: %i, recall %.4f, precision %.4f' % (len(pred_list[0]), tp, re, pre))
    return {
        'tp': tp,
        'precision': pre,
        'recall': re,
    }


def rm_dup_bundle(bundles, size):
    # remember that each bundle need 1 item example index 720 : [1,2,3,4]
    # -> bundle is [720, 1, 2, 3, 4]
    bundles = bundles[:, :size]
    bundles, _ = bundles.sort(dim=1)
    bundles = bundles.unique(dim=0)
    return bundles.tolist()


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

def eval_generated_bundle(gen_bundle, gt_bundle, size_list=[2, 3, 4, 5, 6, 7, 8, 9, 10]):
    gd_bundle = bundle_graph2list(gt_bundle)
    bundle_size_stat = np.array(gt_bundle.sum(axis=1).A.ravel(), dtype=int).squeeze()
    bundle_stat_dict = {}

    # bundle size statistic
    for i in bundle_size_stat:
        if i not in bundle_stat_dict.keys():
            bundle_stat_dict[i] = 1
        else:
            bundle_stat_dict[i] += 1
    # print(bundle[720])

    print(f'evaluating num of bundles: {len(gd_bundle)}')
    print(bundle_stat_dict)

    # eval k-size bundle
    bundle_k_dict = {}
    for i in size_list:
        bundle_k_dict[i] = rm_dup_bundle(gen_bundle, size=i)

    for i in size_list:
        metrics(bundle_k_dict[i], gd_bundle)
    pass


if __name__ == '__main__':
    torch.manual_seed(2023)
    np.random.seed(2023)

    args = get_arg()
    conf = load_config(args.dataset)
    dataset = Datasets(conf)
    item_feat = torch.load(f'datasets/{args.dataset}/self_trained_feature.pt', )

    model = ZeGe(item_feat, conf, dataset.iui_graph).to(conf['device'])
    optimizer = Adam(params=model.parameters(), lr=conf['lr'])
    train_gen_bundle(model, optimizer, dataset.item_item_loader, conf, dataset.bi_graph)
