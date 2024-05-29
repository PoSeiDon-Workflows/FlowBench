""" Benchmarks of node-level Anomoly detection in PyGOD.

"""


import argparse
import os.path as osp
import warnings
from random import choice

import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import MLP
from tqdm import tqdm

from flowbench.dataset import FlowDataset
from flowbench.metrics import (eval_average_precision, eval_precision_at_k,
                               eval_recall_at_k, eval_roc_auc)
from flowbench.unsupervised.pygod import (ANOMALOUS, CONAD, DOMINANT, DONE,
                                          GAAN, GAE, GUIDE, SCAN, AdONE,
                                          AnomalyDAE, Radar)


def init_model(args):
    r""" Initiate model for PyGOD

    Args:
        args (dict): Args from argparser.

    Returns:
        object: Model object.
    """

    # from sklearn.ensemble import IsolationForest
    if not isinstance(args, dict):
        args = vars(args)
    dropout = [0, 0.1, 0.3]
    lr = [0.1, 0.05, 0.01]
    weight_decay = 0.01

    if args['dataset'] == 'inj_flickr':
        # sampling and minibatch training on large dataset flickr
        batch_size = 64
        num_neigh = 3
        epoch = 2
    else:
        batch_size = 0
        num_neigh = -1
        epoch = 300

    model_name = args['model']
    gpu = args['gpu']

    # if hasattr(args, 'epoch'):
    epoch = args.get('epoch', 200)

    if args['dataset'] == 'reddit':
        # for the low feature dimension dataset
        hid_dim = [32, 48, 64]
    else:
        hid_dim = [32, 64, 128, 256]

    if args['dataset'][:3] == 'inj':
        # auto balancing on injected dataset
        alpha = [None]
    else:
        alpha = [0.8, 0.5, 0.2]

    if model_name == "adone":
        return AdONE(hid_dim=choice(hid_dim),
                     weight_decay=weight_decay,
                     dropout=choice(dropout),
                     lr=choice(lr),
                     epoch=epoch,
                     gpu=gpu,
                     batch_size=batch_size,
                     num_neigh=num_neigh)
    elif model_name == 'anomalydae':
        hd = choice(hid_dim)
        return AnomalyDAE(embed_dim=hd,
                          out_dim=hd,
                          weight_decay=weight_decay,
                          dropout=choice(dropout),
                          theta=choice([10., 40., 90.]),
                          eta=choice([3., 5., 8.]),
                          lr=choice(lr),
                          epoch=epoch,
                          gpu=gpu,
                          alpha=choice(alpha),
                          batch_size=batch_size,
                          num_neigh=num_neigh)
    elif model_name == 'conad':
        return CONAD(hid_dim=choice(hid_dim),
                     weight_decay=weight_decay,
                     dropout=choice(dropout),
                     lr=choice(lr),
                     epoch=epoch,
                     gpu=gpu,
                     alpha=choice(alpha),
                     batch_size=batch_size,
                     num_neigh=num_neigh)
    elif model_name == 'dominant':
        return DOMINANT(hid_dim=choice(hid_dim),
                        weight_decay=weight_decay,
                        dropout=choice(dropout),
                        lr=choice(lr),
                        epoch=epoch,
                        gpu=gpu,
                        alpha=choice(alpha),
                        batch_size=batch_size,
                        num_neigh=num_neigh)
    elif model_name == 'done':
        return DONE(hid_dim=choice(hid_dim),
                    weight_decay=weight_decay,
                    dropout=choice(dropout),
                    lr=choice(lr),
                    epoch=epoch,
                    gpu=gpu,
                    batch_size=batch_size,
                    num_neigh=num_neigh)
    elif model_name == 'gaan':
        return GAAN(noise_dim=choice([8, 16, 32]),
                    hid_dim=choice(hid_dim),
                    weight_decay=weight_decay,
                    dropout=choice(dropout),
                    lr=choice(lr),
                    epoch=epoch,
                    gpu=gpu,
                    alpha=choice(alpha),
                    batch_size=batch_size,
                    num_neigh=num_neigh)
    elif model_name == 'gcnae':
        return GAE(hid_dim=choice(hid_dim),
                   weight_decay=weight_decay,
                   dropout=choice(dropout),
                   lr=choice(lr),
                   epoch=epoch,
                   gpu=gpu,
                   batch_size=batch_size,
                   num_neigh=num_neigh)
    elif model_name == 'guide':
        return GUIDE(a_hid=choice(hid_dim),
                     s_hid=choice([4, 5, 6]),
                     weight_decay=weight_decay,
                     dropout=choice(dropout),
                     lr=choice(lr),
                     epoch=epoch,
                     gpu=gpu,
                     alpha=choice(alpha),
                     batch_size=batch_size,
                     num_neigh=num_neigh,
                     cache_dir='./tmp')
    elif model_name == "mlpae":
        return GAE(hid_dim=choice(hid_dim),
                   weight_decay=weight_decay,
                   backbone=MLP,
                   dropout=choice(dropout),
                   lr=choice(lr),
                   epoch=epoch,
                   gpu=gpu,
                   batch_size=batch_size)
    elif model_name == 'radar':
        return Radar(lr=choice(lr), gpu=gpu)
    elif model_name == 'anomalous':
        return ANOMALOUS(lr=choice(lr), gpu=gpu)
    elif model_name == 'scan':
        return SCAN(eps=choice([0.3, 0.5, 0.8]), mu=choice([2, 5, 10]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="1000genome_new_2022",
                        help="Workflow name.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU Index. Default: 0. -1 for CPU only.")
    parser.add_argument("--model", type=str, default="radar",
                        help="supported model: [lof, if, mlpae, scan, radar, "
                             "anomalous, gcnae, dominant, done, adone, "
                             "anomalydae, gaan, guide, conad]. "
                             "Default: dominant")

    args = parser.parse_args()
    model = init_model(args)

    ROOT = osp.join(osp.expanduser("~"), "tmp", "data", args.dataset)
    if args.dataset not in ["cora", "citeseer", "pubmed"]:
        dataset = FlowDataset(root=ROOT,
                              name=args.dataset,
                              node_level=True,
                              binary_labels=True,
                              force_reprocess=False)
    else:
        # NOTE: For debug only. Take standard datasets from PyG.
        dataset = Planetoid(ROOT, args.dataset)
    data = dataset[0]
    auc, ap, prec, rec = [], [], [], []

    num_trials = 10
    try:
        for _ in tqdm(range(num_trials), desc=args.model):
            if args.model == "if" or args.model == "lof":
                model.fit(data.x)
                score = model.decision_function(data.x)
            else:
                # DEBUG: GPU memory issue with local
                model.fit(data)
                score = model.decision_scores_

            y = data.y.bool()
            k = sum(y)
            if np.isnan(score).any():
                warnings.warn('contains NaN, skip one trial.')
                # continue

            auc.append(eval_roc_auc(y, score))
            ap.append(eval_average_precision(y, score))
            prec.append(eval_precision_at_k(y, score, k))
            rec.append(eval_recall_at_k(y, score, k))

        print(f"{args.dataset}",
              f"{model.__class__.__name__:<15}",
              f"AUC: {np.mean(auc):.3f}±{np.std(auc):.3f} ({np.max(auc):.3f})",
              f"AP: {np.mean(ap):.3f}±{np.std(ap):.3f} ({np.max(ap):.3f})",
              f"Prec(K) {np.mean(prec):.3f}±{np.std(prec):.3f} ({np.max(prec):.3f})",
              f"Recall(K): {np.mean(rec):.3f}±{np.std(rec):.3f} ({np.max(rec):.3f})")
    except Exception as e:
        print(f"{args.dataset}",
              f"{model.__class__.__name__:<15}",
              "ERROR:", e)
