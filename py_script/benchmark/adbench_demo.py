""" This is the adbench methods runner, providing complementary methods to PyOD and PyGOD.

LICENSE: MIT
AUTHORS: PoSeiDon Team
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

# unsupervised
from adbench.baseline.DAGMM.run import DAGMM

# semi-supervised
from adbench.baseline.GANomaly.run import GANomaly
from adbench.baseline.DeepSAD.src.run import DeepSAD
from adbench.baseline.REPEN.run import REPEN
from adbench.baseline.PReNet.run import PReNet
from adbench.baseline.DevNet.run import DevNet
from adbench.baseline.FEAWAD.run import FEAWAD

# supervised
from adbench.baseline.FTTransformer.run import FTTransformer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="1000genome_new_2022",
                        help="Workflow name.")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="GPU Index. Default: 0. -1 for CPU only.")
    # parser.add_argument("--model", type=str, default="mlp",
    #                     help="supported model: ['mlp', 'rf', 'dt', 'gcn', 'graphsage']. "
    #                          "Default: 'mlp'")
    parser.add_argument("--setting", type=str, default="unsupervised",
                        help="Setting: unsupervised, semi-supervised, supervised")

    args = vars(parser.parse_args())

    pre_transform = T.Compose([MinMaxNormalizeFeatures(),
                               T.ToUndirected(),
                               T.RandomNodeSplit(split="train_rest",
                                                 num_val=0.2,
                                                 num_test=0.2)])

    WORKFLOW = args['dataset']
    ROOT = osp.join('/tmp', 'data', WORKFLOW)

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

    setting = args["setting"]

    if setting == "unsupervised":
        model = DAGMM(seed=12345, model_name="DAGMM")
        model.fit(Xs[train_mask], ys[train_mask])
        res = model.predict_score(Xs[train_mask], Xs[test_mask])
        pred_y = res >= 0.5
        print("DAGMM", eval_metrics(ys[test_mask], pred_y))

    elif setting == "semi-supervised":
        model_dict = {'GANomaly': GANomaly,
                      'DeepSAD': DeepSAD,
                      'REPEN': REPEN,
                      'DevNet': DevNet,
                      'PReNet': PReNet,
                      'FEAWAD': FEAWAD}
        for mn in ["GANomaly", "DeepSAD", "REPEN", "PReNet"]:
            # for mn in ["FEAWAD"]:
            model = model_dict[mn](seed=12345, model_name=mn)
            model.fit(Xs[train_mask], ys[train_mask])
            res = model.predict_score(Xs[test_mask])
            pred_y = res >= 0.5
            print(mn, eval_metrics(ys[test_mask], pred_y))

    elif setting == "supervised":
        model = FTTransformer(seed=12345, model_name="FTTransformer")
        model.fit(Xs[train_mask], ys[train_mask])
        res = model.predict_score(Xs[test_mask])
        pred_y = res >= 0.5
        print("FTT", eval_metrics(ys[test_mask], pred_y))
