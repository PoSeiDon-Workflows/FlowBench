""" Benchmarks of node-level Anomoly detection in PyGOD.

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
from tqdm import tqdm

from flowbench import list_workflows
from flowbench.dataset import FlowDataset
from flowbench.metrics import (eval_accuracy, eval_average_precision,
                               eval_precision_at_k, eval_recall,
                               eval_recall_at_k, eval_roc_auc)
from flowbench.transforms import MinMaxNormalizeFeatures
from flowbench.unsupervised.pygod import (ANOMALOUS, CONAD, DOMINANT, DONE,
                                          GAAN, GAE, GUIDE, SCAN, AdONE,
                                          AnomalyDAE, Radar)


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
    with open("hps/unsupervised/pygod_config.yaml", 'r') as config_file:
        try:
            config = yaml.safe_load(config_file)
        except yaml.YAMLError as exc:
            print(exc)
    return config[model_name]['default']


models = {
    "adone": AdONE,
    "anomalous": ANOMALOUS,
    "anomalydae": AnomalyDAE,
    "conad": CONAD,
    "dominant": DOMINANT,
    "done": DONE,
    "gaan": GAAN,
    "gae": GAE,
    "guide": GUIDE,
    "radar": Radar,
    "scan": SCAN,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="1000genome")
    parser.add_argument("--model", type=str, default="gae")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--seed", action="store_true")
    args = parser.parse_args()

    if args.seed:
        random_state = 12345
        np.random.seed(random_state)
        torch.manual_seed(random_state)

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

        Xs = ds.x.numpy()
        ys = ds.y.numpy()
        train_mask = ds.train_mask.numpy()
        val_mask = ds.val_mask.numpy()
        test_mask = ds.test_mask.numpy()

        train_Xs, train_ys = Xs[train_mask], ys[train_mask]
        val_Xs, val_ys = Xs[val_mask], ys[val_mask]
        test_Xs, test_ys = Xs[test_mask], ys[test_mask]

        auc, ap, prec, rec = [], [], [], []

        try:
            for i, (clf_name, clf) in enumerate(models.items()):

                model_params = load_params(clf_name)
                # quick test
                if 'epoch' in model_params:
                    model_params['epoch'] = 1
                if 'gpu' in model_params:
                    model_params['gpu'] = args.gpu
                model = clf(**model_params)

                tic = time()
                model.fit(ds[0])
                toc = time()
                score = model.decision_score_

                k = test_ys.sum()
                if np.isnan(score).any():
                    warnings.warn('contains NaN, skip one trial.')
                    # continue

                test_ys_pred = model.label_.numpy()[test_mask]

                acc = eval_accuracy(test_ys, test_ys_pred)
                ap = eval_average_precision(test_ys, test_ys_pred)
                auc = eval_roc_auc(test_ys, test_ys_pred)
                recall = eval_recall(test_ys, test_ys_pred)
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
        except Exception as e:
            print(f"{args.dataset}",
                  f"{model.__class__.__name__:<15}",
                  "ERROR:", e)
