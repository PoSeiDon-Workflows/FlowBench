""" Benchmarks of node-level Anomoly detection in PyGOD.

"""
import argparse
import os.path as osp
import warnings

import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
from numpy import percentile
from torch_geometric.datasets import Planetoid

from flowbench.dataset import FlowDataset
from flowbench.metrics import (eval_average_precision, eval_precision_at_k,
                               eval_recall_at_k, eval_roc_auc)
from flowbench.unsupervised.pyod import (ABOD, CBLOF, GMM, HBOS, INNE, KDE,
                                         KNN, LMDD, LOF, LSCP, MCD, OCSVM, PCA,
                                         FeatureBagging, IForest)


# TODO: add sklearnex to accelerate sklearn
# from sklearnex import patch_sklearn
# patch_sklearn()

warnings.filterwarnings("ignore")

random_state = 12345
detector_list = [LOF(n_neighbors=5, n_jobs=-1), LOF(n_neighbors=10, n_jobs=-1), LOF(n_neighbors=15, n_jobs=-1),
                 LOF(n_neighbors=20, n_jobs=-1), LOF(n_neighbors=25, n_jobs=-1), LOF(n_neighbors=30, n_jobs=-1),
                 LOF(n_neighbors=35, n_jobs=-1), LOF(n_neighbors=40, n_jobs=-1), LOF(n_neighbors=45, n_jobs=-1),
                 LOF(n_neighbors=50, n_jobs=-1)]

classifiers = {
    'Angle-based Outlier Detector (ABOD)':
        ABOD(),
    'Cluster-based Local Outlier Factor (CBLOF)':
        CBLOF(check_estimator=False, random_state=random_state, n_jobs=-1),
    'Feature Bagging':
        FeatureBagging(LOF(n_neighbors=35), random_state=random_state, n_jobs=-1),
    'Histogram-base Outlier Detection (HBOS)':
        HBOS(),
    'Isolation Forest':
        IForest(random_state=random_state, n_jobs=-1),
    'K Nearest Neighbors (KNN)':
        KNN(n_jobs=-1),
    'Average KNN (AKNN)':
        KNN(method='mean', n_jobs=-1),
    'Local Outlier Factor (LOF)':
        LOF(n_neighbors=35, n_jobs=-1),
    'Minimum Covariance Determinant (MCD)':
        MCD(random_state=random_state),
    'One-class SVM (OCSVM)':
        OCSVM(),
    'Principal Component Analysis (PCA)':
        PCA(random_state=random_state),
    'Locally Selective Combination (LSCP)':
        LSCP(detector_list, random_state=random_state),
    'Isolation-based anomaly detection using nearest-neighbor ensembles (INNE)':
        INNE(max_samples=2, random_state=random_state),
    'Gaussian Mixture Model (GMM)':
        GMM(random_state=random_state),
    'Kernel Density Estimation (KDE)':
        KDE(),
    'Linear Method for Deviation-based Outlier Detection (LMDD)':
        LMDD(random_state=random_state),
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='1000genome_new_2022')
    args = parser.parse_args()

    # ROOT = osp.join(osp.expanduser("~"), "tmp", "data", args.dataset)
    ROOT = osp.join("/tmp", "data", "poseidon")
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

    Xs = dataset.data.x.numpy()
    ys = dataset.data.y.numpy()
    print('Number of inliers: %i' % (ys == 0).sum())
    print('Number of outliers: %i' % (ys == 1).sum())
    print('Ground truth shape is {shape}. Outlier are1  and inliers are 0.\n'.format(shape=len(ys)))

    for i, (clf_name, clf) in enumerate(classifiers.items()):
        print(i + 1, clf_name, end="\t")

        auc, ap, prec, rec = [], [], [], []
        for _ in range(1):
            clf.fit(Xs)
            scores_pred = clf.decision_function(Xs) * -1
            y_pred = clf.predict(Xs)
            k = sum(ys)
            threshold = percentile(scores_pred, 100 * (ys == 1).sum() / len(ys))
            n_errors = (y_pred != ys).sum()
            # print(np.bincount(y_pred[y_pred != ys]), "n_error: ", n_errors, "OOD ratio", f"{n_errors / len(ys):.4f}")

            auc.append(eval_roc_auc(ys, y_pred))
            ap.append(eval_average_precision(ys, y_pred))
            prec.append(eval_precision_at_k(ys, y_pred, k))
            rec.append(eval_recall_at_k(ys, y_pred, k))
        print(f"{args.dataset}",
              f"{clf_name}\n",
              f"AUC: {np.mean(auc):.3f}±{np.std(auc):.3f} ({np.max(auc):.3f})",
              f"AP: {np.mean(ap):.3f}±{np.std(ap):.3f} ({np.max(ap):.3f})",
              f"Prec(K) {np.mean(prec):.3f}±{np.std(prec):.3f} ({np.max(prec):.3f})",
              f"Recall(K): {np.mean(rec):.3f}±{np.std(rec):.3f} ({np.max(rec):.3f})")
