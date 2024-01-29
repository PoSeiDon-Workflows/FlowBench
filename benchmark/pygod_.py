""" Benchmarks of node-level Anomoly detection in PyGOD.

"""


import argparse
import os.path as osp
import warnings

import numpy as np
from torch_geometric.datasets import Planetoid
from tqdm import tqdm

from flowbench.dataset import FlowDataset
from flowbench.metrics import (eval_average_precision, eval_precision_at_k,
                               eval_recall_at_k, eval_roc_auc)
from flowbench.utils import init_model

# torch.manual_seed(12345)
# np.random.seed(12345)
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
