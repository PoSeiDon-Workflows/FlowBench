""" Dataset Cartography

* identify the `easy-to-learn`, `hard-to-learn`, and `ambiguous` samples
* clean the data using `CleanLab`
"""

# %% Import packages
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
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score, f1_score)
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier
from torch.nn import CrossEntropyLoss, Linear, Module, ReLU, Softmax
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn.models import MLP
from tqdm.notebook import tqdm

torch.manual_seed(0)
np.random.seed(0)

# %%
WORKFLOW = "montage"
ROOT = osp.join(osp.expanduser("~"), "tmp", "data", WORKFLOW)
pre_transform = T.Compose([MinMaxNormalizeFeatures(),
                           T.ToUndirected(),
                           T.RandomNodeSplit(split="train_rest",
                                             num_val=0.2,
                                             num_test=0.2)])
ds_node = PSD_Dataset(root=ROOT,
                      name=WORKFLOW,
                      node_level=True,
                      binary_labels=True,
                      pre_transform=pre_transform)

data = ds_node[0]

Xs = data.x.reshape(-1, ds_node.num_nodes_per_graph, ds_node.num_features)
ys = data.y.reshape(-1, ds_node.num_nodes_per_graph)

# %% [select a single job]

job_id = 44
Xs_job = Xs[:, job_id, :]
ys_job = ys[:, job_id]
data = Data(x=Xs_job, y=ys_job)
# random split
T.RandomNodeSplit(split="random", num_val=0.2, num_test=0.2)(data)
model = MLP(in_channels=ds_node.num_features,
            hidden_channels=128,
            num_layers=3,
            out_channels=2, dropout=0.5)

optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# %% [START with tabular data]
model.reset_parameters()
model.train()
early_stopping = 0
best = 1e10
pbar = tqdm(range(1000), desc=f"MLP(X_{job_id})", leave=True)
train_accs, val_accs = [], []
train_losses, val_losses = [], []
init_patient = 100
patient = 100
for e in pbar:
    optimizer.zero_grad()
    X_output = model(data.x[data.train_mask])
    train_loss = F.cross_entropy(X_output, data.y[data.train_mask])
    train_loss.backward()
    optimizer.step()
    train_acc = (X_output.argmax(1) == data.y[data.train_mask]).sum() / X_output.shape[0]

    val_loss = F.cross_entropy(model(data.x[data.val_mask]), data.y[data.val_mask])
    val_acc = (model(data.x[data.val_mask]).argmax(1) == data.y[data.val_mask]).sum() / data.x[data.val_mask].shape[0]
    # print(loss.item(), acc.item(), val_loss.item(), val_acc.item(), early_stopping)
    pbar.set_postfix({"train_loss": train_loss.item(),
                      "train_acc": train_acc.item(),
                      "val_loss": val_loss.item(),
                      "val_acc": val_acc.item(),
                      "counting": patient})
    train_accs.append(train_acc.item())
    val_accs.append(val_acc.item())
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())
    if val_loss <= best:
        best = val_loss
        patient = init_patient
    else:
        # early_stopping = 0
        patient -= 1

    if patient <= 0:
        print("Early stopping")
        break


model.eval()
test_output = model(data.x[data.test_mask])
test_pred = test_output.argmax(1)
# acc = (X_output.argmax(1) == data.y[data.test_mask]).sum() / X_output.shape[0]
test_res = eval_metrics(data.y[data.test_mask].numpy(), test_pred.detach().numpy())
print(test_res)

# %% [Visualize the dataset Cartography]

mlp = MLP(in_channels=25, hidden_channels=128, out_channels=2, num_layers=2)

optimizer = Adam(mlp.parameters(), lr=1e-4, weight_decay=1e-5)
loss_func = CrossEntropyLoss()
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
    # print(train_acc)
    X_probs.append(X_prob.detach().cpu().numpy()[np.arange(n_samples), y_].reshape(-1, 1))
# %%
print(train_acc)
res = np.hstack(X_probs)
fig = plt.figure(figsize=(6, 4), )
gs = fig.add_gridspec(2, 3, height_ratios=[5, 1])
ax0 = fig.add_subplot(gs[0, :])

conf = res[:, 100:].mean(1)
vari = res[:, 100:].std(1)
corr = (res[:, 100:] >= 0.5).sum(1)
# plt.scatter(vari, conf)
pal = sns.diverging_palette(260, 15, sep=10, center="dark")
plot = sns.scatterplot(x=vari, y=conf, ax=ax0, palette=pal, hue=corr / 100, style=corr / 100)


def bb(c):
    return dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")


an1 = ax0.annotate("ambiguous", xy=(0.9, 0.5), xycoords="axes fraction", fontsize=12, color='black',
                   va="center", ha="center", rotation=360, bbox=bb('black'))
an2 = ax0.annotate("easy-to-learn", xy=(0.27, 0.85), xycoords="axes fraction", fontsize=12, color='black',
                   va="center", ha="center", bbox=bb('r'))
an3 = ax0.annotate("hard-to-learn", xy=(0.27, 0.25), xycoords="axes fraction", fontsize=12, color='black',
                   va="center", ha="center", bbox=bb('b'))
plot.legend(fancybox=True, shadow=True, ncol=1)

plot.set_xlabel('variability')
plot.set_ylabel('confidence')

plot.set_title("Data Map", fontsize=12)
fig.tight_layout()
# %% [CleanLab]
mlp_skorch = NeuralNetClassifier(mlp)
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
predicted_labels = pred_probs.argmax(axis=1)
acc = accuracy_score(y_.numpy(), predicted_labels)
roc_auc = roc_auc_score(y_.numpy(), predicted_labels)
f1 = f1_score(y_.numpy(), predicted_labels)
print(f"Cross-validated estimate of accuracy on held-out data: {acc:.4f} {f1:.4f} {roc_auc:.4f}")
# %%
from scipy.special import softmax
ranked_label_issues = find_label_issues(
    y_.numpy(),
    softmax(pred_probs, axis=1),
    return_indices_ranked_by="self_confidence",
)

print(f"Cleanlab found {len(ranked_label_issues)} label issues.")
print(f"Top 15 most likely label errors: \n {ranked_label_issues[:15]}")
# %%
