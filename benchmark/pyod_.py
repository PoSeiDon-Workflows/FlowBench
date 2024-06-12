""" Benchmarks of node-level Anomoly detection in PyOD.

Author: PoSeiDon Team
License: MIT
"""
import argparse
import os.path as osp
import warnings
from time import time

import numpy as np
import torch
import torch_geometric.transforms as T
import yaml

from flowbench import list_workflows
from flowbench.dataset import FlowDataset
from flowbench.metrics import (eval_accuracy, eval_average_precision,
                               eval_precision_at_k, eval_recall,
                               eval_recall_at_k, eval_roc_auc)
from flowbench.transforms import MinMaxNormalizeFeatures
from flowbench.unsupervised.pyod import (ABOD, CBLOF, GMM, HBOS, INNE, KDE,
                                         KNN, LMDD, LOF, MCD, OCSVM, PCA,
                                         FeatureBagging, IForest)

warnings.filterwarnings("ignore")


def load_workflow(workflow="1000genome", root="/tmp"):
    r""" Load workflow dataset.

    Args:
        workflow (str): Name of the workflow dataset. Default: "1000genome".

    Returns:
        pyg.data.Dataset: The dataset.
    """

    ROOT = osp.join(root, "data", "flowbench")
    pre_transform = T.Compose([MinMaxNormalizeFeatures(),
                               T.ToUndirected(),
                               T.RandomNodeSplit(split="train_rest",
                                                 num_val=0.1,
                                                 num_test=0.2)])
    ds = FlowDataset(root=ROOT,
                     name=workflow,
                     binary_labels=True,
                     node_level=True,
                     pre_transform=pre_transform,
                     force_reprocess=False)
    return ds


def load_params(model_name):
    r""" Load default hyperparameters for the model from preconfigured yaml file.

    Args:
        model_name (str): Name of the model.

    Returns:
        dict: Hyperparameters for the model.
    """
    with open("hps/unsupervised/pyod_config.yaml", 'r') as config_file:
        try:
            config = yaml.safe_load(config_file)
        except yaml.YAMLError as exc:
            print(exc)
    return config[model_name]['default']


models = {
    "abod": ABOD,
    "cblof": CBLOF,
    "feature_bagging": FeatureBagging,
    "gmm": GMM,
    "hbos": HBOS,
    "iforest": IForest,
    "inne": INNE,
    "kde": KDE,
    "knn": KNN,
    "lmdd": LMDD,
    "lof": LOF,
    "mcd": MCD,
    "ocsvm": OCSVM,
    "pca": PCA,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all')
    parser.add_argument('--model', type=str, default='gmm')
    parser.add_argument('--seed', action='store_true')
    args = parser.parse_args()

    if args.seed:
        random_state = 12345
        torch.manual_seed(random_state)
        np.random.seed(random_state)

    if args.dataset != "all":
        wfs = [args.dataset]
    else:
        wfs = list_workflows()

    if args.model != "all":
        models = {args.model: models[args.model]}
    else:
        models = models

    for wf in wfs:
        ds = load_workflow(wf)

        Xs = ds.data.x.numpy()
        ys = ds.data.y.numpy()

        train_mask = ds.train_mask.numpy()
        val_mask = ds.val_mask.numpy()
        test_mask = ds.test_mask.numpy()

        train_Xs, train_ys = Xs[train_mask], ys[train_mask]
        val_Xs, val_ys = Xs[val_mask], ys[val_mask]
        test_Xs, test_ys = Xs[test_mask], ys[test_mask]

        for i, (clf_name, clf) in enumerate(models.items()):
            model_params = load_params(clf_name)

            model = clf(**model_params)

            tic = time()
            model.fit(train_Xs)
            toc = time()

            test_ys_pred = model.predict(test_Xs)

            k = test_ys.sum()
            ap = eval_average_precision(test_ys, test_ys_pred)
            auc = eval_roc_auc(test_ys, test_ys_pred)
            recall = eval_recall(test_ys, test_ys_pred)
            acc = eval_accuracy(test_ys, test_ys_pred)
            prec_k = eval_precision_at_k(test_ys, test_ys_pred, k)
            recall_k = eval_recall_at_k(test_ys, test_ys_pred, k)

            print(f"{args.dataset}",
                  f"{args.model}:",
                  f"train_time {toc - tic:.3f}",
                  f"Accuracy {acc:.3f}",
                  f"AP {ap:.3f}",
                  f"AUC {auc:.3f}",
                  f"Recall {recall:.3f}",
                  f"prec_k {prec_k:.3f}",
                  f"recall_k {recall_k:.3f}")
