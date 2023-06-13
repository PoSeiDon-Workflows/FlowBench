""" Benchmarks of node-level anomaly detection with supervised learning

"""
import argparse
import os.path as osp
import random
from time import time

import numpy as np
import torch
import torch_geometric.transforms as T
from psd_gnn.dataset import PSD_Dataset
from psd_gnn.transforms import MinMaxNormalizeFeatures
from psd_gnn.utils import eval_metrics, process_args
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from torch_geometric.loader import DataLoader

torch.manual_seed(12345)
np.random.seed(12345)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="1000genome_new_2022",
                        help="Workflow name.")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="GPU Index. Default: 0. -1 for CPU only.")
    parser.add_argument("--model", type=str, default="mlp",
                        help="supported model: ['mlp', 'rf', 'dt', 'gcn', 'graphsage']. "
                             "Default: 'mlp'")

    args = vars(parser.parse_args())

    pre_transform = T.Compose([MinMaxNormalizeFeatures(),
                               T.ToUndirected(),
                               T.RandomNodeSplit(split="train_rest",
                                                 num_val=0.2,
                                                 num_test=0.2)])

    WORKFLOW = args['dataset']
    ROOT = osp.join(osp.expanduser('~'), 'tmp', 'data', WORKFLOW)

    dataset = PSD_Dataset(root=ROOT,
                          name=WORKFLOW,
                          node_level=True,
                          binary_labels=True,
                          normalize=False,
                          force_reprocess=False,
                          pre_transform=pre_transform)
    data = dataset[0]
    Xs = data.x.numpy()
    ys = data.y.numpy()
    train_mask = data.train_mask.numpy()
    val_mask = data.val_mask.numpy()
    test_mask = data.test_mask.numpy()

    tic = time()
    if args['model'] == 'mlp':
        clf = MLPClassifier(random_state=1, max_iter=300)
        clf.fit(Xs[train_mask], ys[train_mask])
        y_pred = clf.predict(Xs[test_mask])
        res = eval_metrics(y_true=ys[test_mask], y_pred=y_pred)
        print(res)
    elif args['model'] == 'rf':
        clf = RandomForestClassifier(random_state=1)
        clf.fit(Xs[train_mask], ys[train_mask])
        y_pred = clf.predict(Xs[test_mask])
        res = eval_metrics(y_true=ys[test_mask], y_pred=y_pred)
        print(res)
    elif args['model'] == 'dt':
        clf = DecisionTreeClassifier(random_state=1)
        clf.fit(Xs[train_mask], ys[train_mask])
        y_pred = clf.predict(Xs[test_mask])
        res = eval_metrics(y_true=ys[test_mask], y_pred=y_pred)
        print(res)
    toc = time()
    print(f"Time elapsed: {toc - tic:.2f}s")
