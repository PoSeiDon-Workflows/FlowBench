""" An example of using PyOD models in FlowBench.

It takes GMM (Gaussian Mixture Model) as an example to demonstrate
how to use PyOD models in FlowBench, including loading a dataset, training a
model, and evaluating the model.

Author: PoSeiDon Team
License: MIT
"""
import argparse
from time import time

import numpy as np
import torch

from examples.utils import load_workflow
from flowbench.metrics import (eval_accuracy,
                               eval_average_precision,
                               eval_recall,
                               eval_roc_auc)
from flowbench.unsupervised.pyod import GMM

torch.manual_seed(12345)
np.random.seed(12345)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="1000genome")
args = parser.parse_args()
ds = load_workflow(args.dataset)

Xs = ds.x.numpy()
ys = ds.y.numpy()

print(f'Number of normal (0): {(ys == 0).sum()}')
print(f'Number of abnormal (1): {(ys == 1).sum()}')
print(f'anomalous rate: {np.mean(ys):.3f}')
print(f'Ground truth shape: {len(ys)}.\n')


train_mask = ds.train_mask.numpy()
val_mask = ds.val_mask.numpy()
test_mask = ds.test_mask.numpy()

train_Xs, train_ys = Xs[train_mask], ys[train_mask]
val_Xs, val_ys = Xs[val_mask], ys[val_mask]
test_Xs, test_ys = Xs[test_mask], ys[test_mask]


gmm = GMM(n_components=1,
          covariance_type='full',
          tol=1e-3,
          reg_covar=1e-6,
          max_iter=100,
          n_init=1,
          init_params='kmeans',
          random_state=12345)


tic = time()
gmm.fit(train_Xs)
toc = time()
print(f"Training time: {toc - tic:.3f} sec")

tic = time()
test_ys_pred = gmm.predict(test_Xs)
toc = time()
print(f"Inference time: {toc - tic:.3f} sec")

ap = eval_average_precision(test_ys, test_ys_pred)
auc = eval_roc_auc(test_ys, test_ys_pred)
recall = eval_recall(test_ys, test_ys_pred)
acc = eval_accuracy(test_ys, test_ys_pred)

print(f"{args.dataset} - GMM:",
      f"Accuracy {acc:.3f}",
      f"AP {ap:.3f}",
      f"AUC {auc:.3f}",
      f"Recall {recall:.3f}")
