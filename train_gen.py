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


if __name__ == '__main__':
    torch.manual_seed(2023)
    np.random.seed(2023)

    args = get_arg()
    conf = load_config(args.dataset)
    dataset = Datasets(conf)
    item_feat = torch.load(f'datasets/{args.dataset}/self_trained_feature.pt', )

    model = ZeGe(item_feat, conf, dataset.iui_graph).to(conf['device'])
    optimizer = Adam(params=model.parameters(), lr=conf['lr'])
    train_gen_bundle(model, optimizer, dataset.item_item_loader, conf)
