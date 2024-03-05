import os
from argparse import ArgumentParser

import yaml
from torch.optim import Adam
from tqdm import tqdm

from models.BunGen import *
from utility import *


def load_config(dataset):
    f = open('gen.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configs = yaml.safe_load(f)
    config = configs[dataset]
    config['dataset'] = dataset
    config['device'] = device
    return config


def get_arg():
    argp = ArgumentParser()
    argp.add_argument('-d', '--dataset', type=str, default='clothing')
    args = argp.parse_args()
    return args


def train_gen_bundle(model, optimizer, dataloader, conf):
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

        # model.eval()
        # gen_bundle, prob = model.gen(10)
        # print(gen_bundle[881], prob[881])
        # print(gen_bundle[720], prob[720])
        # print(gen_bundle[2294], prob[2294])
        # print(gen_bundle[1933], prob[1933])

    model.eval()
    res_bundle, prob = model.gen(conf['bundle_size'])
    torch.save(res_bundle, os.path.join(conf['data_path'], conf['dataset'], 'bundle.pt'))


def recall(pred, gdtruth):
    pass


def ndcg(pred, gdtruth):
    pass


def precision(pred, gd_truth):
    # pred: list of bundle indices # ex : [[0,1,2], [3,4], ...]
    # gdtruth: one-hot of each bundle in sparse type # ex [[1,0,0,1],[1,1,0,0],...]
    tp = 0
    num_item = gd_truth.shape[1]
    for i in gd_truth:
        for j in pred:
            # print('j:', j)
            j = torch.sparse_coo_tensor(indices=torch.tensor(np.array([j])),
                                        values=torch.ones(len(j)),
                                        size=(num_item,))
            # print('j sparse:', j)
            if torch.sum(torch.abs(i - j)) == 0:
                tp += 1
    return tp / len(pred)


if __name__ == '__main__':
    args = get_arg()
    conf = load_config(args.dataset)
    dataset = Datasets(conf)

    item_feat = torch.load(f'datasets/{args.dataset}/pretrain_feature.pt')
    model = BunGen(item_feat, conf, dataset.iui_graph).to(conf['device'])
    optimizer = Adam(model.parameters())

    train_gen_bundle(model, optimizer, dataset.item_item_loader, conf)
