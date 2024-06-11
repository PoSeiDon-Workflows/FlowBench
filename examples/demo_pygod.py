""" An example of using PyGOD models in FlowBench.

It takes GAE (Graph Auto-Encoder) as an example to demonstrate
how to use PyGOD models in FlowBench, including loading a dataset, training a
model, and evaluating the model.

Author: PoSeiDon Team
License: MIT
"""

import argparse
from time import time

import numpy as np
import torch

from examples.utils import load_workflow
from flowbench.metrics import (eval_accuracy, eval_average_precision,
                               eval_recall, eval_roc_auc)
from flowbench.unsupervised.pygod import GAE

# torch.manual_seed(12345)
# np.random.seed(12345)

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

gae = GAE(hid_dim=64,
          num_layers=4,
          dropout=0,
          weight_decay=0,
          contamination=0.1,
          lr=0.004,
          epoch=100,
          gpu=0,
          )

tic = time()
gae.fit(ds[0])
toc = time()
print(f"Training time: {toc - tic:.3f} sec")

test_ys_pred = gae.label_.numpy()[test_mask]
ap = eval_average_precision(test_ys, test_ys_pred)
auc = eval_roc_auc(test_ys, test_ys_pred)
recall = eval_recall(test_ys, test_ys_pred)
acc = eval_accuracy(test_ys, test_ys_pred)

print(f"{args.dataset} - GAE:",
      f"Accuracy {acc:.3f}",
      f"AP {ap:.3f}",
      f"AUC {auc:.3f}",
      f"Recall {recall:.3f}")
