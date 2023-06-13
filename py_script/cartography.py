""" Dataset Cartography

* identify the `easy-to-learn`, `hard-to-learn`, and `ambiguous` samples
* clean the data using `CleanLab`

Author: PoSeiDon Team
License: MIT
"""

# %% imports

import os.path as osp
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from cleanlab.classification import CleanLearning
from cleanlab.filter import find_label_issues
from psd_gnn.dataset import PSD_Dataset
from psd_gnn.transforms import MinMaxNormalizeFeatures
from psd_gnn.utils import eval_metrics, parse_adj
from scipy.special import softmax
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             roc_auc_score)
from sklearn.model_selection import cross_val_predict, train_test_split
from skorch import NeuralNetClassifier
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn.models import MLP

torch.manual_seed(0)
np.random.seed(0)

# %% load workflow data
WORKFLOW = "1000genome_new_2022"
ROOT = osp.join(osp.expanduser("~"), "tmp", "data", WORKFLOW)
# normalized data
pre_transform = T.Compose([MinMaxNormalizeFeatures(),
                           T.ToUndirected(),
                           T.RandomNodeSplit(split="train_rest",
                                             num_val=0.2,
                                             num_test=0.2)])
node_ds = PSD_Dataset(root=ROOT,
                      name=WORKFLOW,
                      node_level=True,
                      binary_labels=True,
                      pre_transform=pre_transform)

all_data = node_ds[0]

Xs = all_data.x.reshape(-1, node_ds.num_nodes_per_graph, node_ds.num_features)
ys = all_data.y.reshape(-1, node_ds.num_nodes_per_graph)

# %% select a single job to analysis

job_id = 44
Xs_job = Xs[:, job_id, :]
ys_job = ys[:, job_id]
data = Data(x=Xs_job, y=ys_job)
# random split
T.RandomNodeSplit(split="random", num_val=0.2, num_test=0.2)(data)

# %% [Visualize the dataset Cartography]

# build a MLP model for a single job

mlp = MLP(in_channels=node_ds.num_features,
          hidden_channels=128,
          out_channels=2,
          num_layers=3)

optimizer = Adam(mlp.parameters(),
                 lr=1e-4,
                 weight_decay=1e-5)

# a list of predict probabilities for each epoch
X_probs = []
X_ = Xs[:, job_id, :]
y_ = ys[:, job_id]
n_samples = X_.shape[0]
for e in range(200):
    optimizer.zero_grad()
    X_output = mlp(X_)
    X_prob = F.softmax(X_output, dim=1)
    loss = F.cross_entropy(X_prob, y_)
    loss.backward()
    optimizer.step()
    train_acc = ((X_prob.argmax(1) == y_).sum() / X_prob.shape[0]).item()
    # probabilities of true labels
    X_probs.append(X_prob.detach().cpu().numpy()[np.arange(n_samples), y_].reshape(-1, 1))

# %%
print(train_acc)
res = np.hstack(X_probs)
fig = plt.figure(figsize=(5, 3), tight_layout=True)
# gs = fig.add_gridspec(2, 3, height_ratios=[5, 1])
# ax0 = fig.add_subplot(gs[0, :])
ax0 = fig.add_subplot(111)

conf = res[:, 100:].mean(1)
vari = res[:, 100:].std(1)
corr = (res[:, 100:] >= 0.5).sum(1)
# plt.scatter(vari, conf)
pal = sns.diverging_palette(260, 15, sep=10, center="dark")
# plot = sns.scatterplot(x=vari, y=conf, ax=ax0, palette=pal, hue=corr // 20, style=corr // 20)
idx = np.digitize(corr / 100, np.arange(0, 1, 0.2)) - 1
plot = sns.scatterplot(x=vari, y=conf, ax=ax0, palette=pal,
                       hue=np.arange(0, 1, 0.2)[idx].round(1),
                       style=np.arange(0, 1, 0.2)[idx].round(1))


def bb(c):
    return dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")


an1 = ax0.annotate("ambiguous", xy=(0.9, 0.6), xycoords="axes fraction", fontsize=12, color='black',
                   va="center", ha="center", rotation=360, bbox=bb('black'))
an2 = ax0.annotate("easy-to-learn", xy=(0.27, 0.85), xycoords="axes fraction", fontsize=12, color='black',
                   va="center", ha="center", bbox=bb('r'))
an3 = ax0.annotate("hard-to-learn", xy=(0.27, 0.25), xycoords="axes fraction", fontsize=12, color='black',
                   va="center", ha="center", bbox=bb('b'))
plot.legend(fancybox=True, shadow=True, ncol=1)

plot.set_xlabel('Variability')
plot.set_ylabel('Confidence')

# plot.set_title("Data Map", fontsize=12)
fig.tight_layout()
fig.savefig("data_map.pdf")

# %% [CleanLab]
mlp_skorch = NeuralNetClassifier(mlp, verbose=0)
num_crossval_folds = 10
pred_probs = cross_val_predict(
    mlp_skorch,
    # model,
    # StandardScaler().fit_transform(Xs).astype("float32"),
    X_.numpy(),
    y_.numpy(),
    cv=num_crossval_folds,
    method="predict_proba",
    n_jobs=-1,
    verbose=0,
)


# %%

ranked_label_issues = find_label_issues(
    y_.numpy(),
    softmax(pred_probs, axis=1),
    return_indices_ranked_by="self_confidence",
)

print(f"Cleanlab found {len(ranked_label_issues)} label issues.")
print(f"Top 15 most likely label errors: \n {ranked_label_issues[:15]}")
# %%
# from cleanlab.dataset import health_summary
# health_summary(y_.numpy(), pred_probs, class_names=mlp)
# %%
y_probs = softmax(pred_probs, axis=1)[np.arange(pred_probs.shape[0]), y_.numpy()]
true_idx = np.where(y_probs > 0.5)[0]
false_idx = np.array([x for x in np.where(y_probs < 0.5)[0] if x not in ranked_label_issues])

# plt.figure(figsize=(5, 4), tight_layout=True)
# sns.scatterplot(x=true_idx, y=y_probs[true_idx], label="Correct Pred. Samples")
# sns.scatterplot(x=false_idx, y=y_probs[false_idx], label="Incorrect Pred. Samples")
# sns.scatterplot(x=ranked_label_issues, y=y_probs[ranked_label_issues], label="OOD samples")
# plt.xlabel("Sample Index")
# plt.ylabel("Predicted Probability")
# %%
df = pd.DataFrame(y_probs, columns=['y_probs'])
cat = np.empty(df.shape[0], dtype="U9")
cat[true_idx] = "Correct"
cat[false_idx] = "Incorrect"
cat[ranked_label_issues] = "OOD"
df['Prediction'] = cat
df['Label'] = data.y.numpy()

plt.figure(figsize=(5, 3), tight_layout=True)
sns.scatterplot(data=df, x=np.arange(351), y="y_probs", hue="Prediction", style="Label")
plt.ylabel("Predicted Probablity of True Label")
plt.xlabel("Sample index")
plt.savefig("label_analysis.pdf")

# %%
# clean data by removing the OOD data

# %%
predicted_labels = pred_probs.argmax(axis=1)
acc = accuracy_score(y_.numpy(), predicted_labels)
roc_auc = roc_auc_score(y_.numpy(), predicted_labels)
f1 = f1_score(y_.numpy(), predicted_labels)
print(f"Cross-validated estimate of accuracy on held-out data: {acc:.4f} {f1:.4f} {roc_auc:.4f}")

print("before cleaning")
print(classification_report(y_.numpy(), predicted_labels))

cleaned_X = np.delete(X_, ranked_label_issues, axis=0)
cleaned_y = np.delete(y_, ranked_label_issues, axis=0)

cleaned_pred_probs = cross_val_predict(
        mlp_skorch,
        # StandardScaler().fit_transform(cleaned_X).astype("float32"),
        cleaned_X,
        cleaned_y,
        cv=num_crossval_folds,
        method="predict_proba",
        n_jobs=-1,
    )

cleaned_pred_labels = cleaned_pred_probs.argmax(axis=1)
acc = accuracy_score(cleaned_y, cleaned_pred_labels)
roc_auc = roc_auc_score(cleaned_y.numpy(), cleaned_pred_labels)
f1 = f1_score(cleaned_y.numpy(), cleaned_pred_labels)
print(f"Cross-validated estimate of accuracy on held-out data: {acc:.4f} {f1:.4f} {roc_auc:.4f}")
print("after cleaning")
print(classification_report(cleaned_y, cleaned_pred_labels))
# new_job_acc.append(acc)



# %%
