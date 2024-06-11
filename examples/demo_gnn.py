""" """
import os.path as osp

import lightning as L
import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader, NeighborLoader, RandomNodeLoader
from torch_geometric.nn.models import GCN
from tqdm import tqdm

from examples.utils import load_workflow
from flowbench.dataset import FlowDataset
from flowbench.supervised.gnn import GNN, GNN_v2
from flowbench.transforms import MinMaxNormalizeFeatures
import time
import argparse
from flowbench.metrics import eval_accuracy, eval_f1, eval_average_precision, eval_recall, eval_roc_auc

torch.set_float32_matmul_precision("medium")
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="1000genome")
args = parser.parse_args()

ds = load_workflow(workflow=args.dataset)
# train_mask = ds[0].train_mask
# val_mask = ds[0].val_mask
# test_mask = ds[0].test_mask
# # print(ds.y.bincount())
# # exit()
# # RandomNodeLoader()
# train_loader = NeighborLoader(ds[0], num_neighbors=[-1, -1], batch_size=2048, shuffle=True, input_nodes=train_mask)
# val_loader = NeighborLoader(ds[0], num_neighbors=[-1, -1], batch_size=2048, shuffle=False, input_nodes=val_mask)
# test_loader = NeighborLoader(ds[0], num_neighbors=[-1, -1], batch_size=2048, shuffle=False, input_nodes=test_mask)

# # loader = DataLoader(ds, batch_size=256, shuffle=True)
# model = GNN_v2(ds.num_node_features, ds.num_classes, num_layers=4, lr=1e-3)
# print(model)
# trainer = L.Trainer(max_epochs=200, logger=False, accelerator="gpu", devices=1)
# tic = time.time()
# trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
# toc = time.time()
# print(f"Training time: {toc - tic}")
# trainer.test(model, dataloaders=test_loader)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
ds = ds.to(DEVICE)
# loader = DataLoader(ds, batch_size=256, shuffle=True)
train_loader = NeighborLoader(ds[0], num_neighbors=[-1, -1], batch_size=512, shuffle=True, input_nodes=ds[0].train_mask)
test_loader = NeighborLoader(ds[0], num_neighbors=[-1, -1], batch_size=512, shuffle=False, input_nodes=ds[0].test_mask)
model = GCN(ds.num_node_features, ds.num_classes, num_layers=4)
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

tic = time.time()
pbar = tqdm(range(200))
for epoch in pbar:
    model.train()
    total = 0
    correct = 0
    for data in train_loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(out.data, 1)
        total += data.y.size(0)
        correct += (predicted == data.y).sum().item()

    # accuracy = 100 * correct / total
    # pbar.set_postfix({"loss": f"{loss.item():.4f}", "accuracy": f"{accuracy:.2f}%"})

toc = time.time()

# eval on test, with accuracy, recall, roc_auc, f1, precision
model.eval()
total = 0
correct = 0
y_true = []
y_pred = []
for data in test_loader:
    data = data.to(DEVICE)
    out = model(data.x, data.edge_index)
    _, predicted = torch.max(out.data, 1)
    total += data.y.size(0)
    correct += (predicted == data.y).sum().item()
    y_true.append(data.y)
    y_pred.append(predicted)
accuracy = 100 * correct / total
y_true = torch.cat(y_true, dim=0).detach().cpu().numpy()
y_pred = torch.cat(y_pred, dim=0).detach().cpu().numpy()

acc = eval_accuracy(y_true, y_pred)
f1 = eval_f1(y_true, y_pred)
auc = eval_roc_auc(y_true, y_pred)
ap = eval_average_precision(y_true, y_pred)
recall = eval_recall(y_true, y_pred)
print(f"{args.dataset} - GNN:",
      f"Training time {toc - tic:.3f}",
      f"Accuracy {acc:.3f}",
      f"AP {ap:.3f}",
      f"AUC {auc:.3f}",
      f"Recall {recall:.3f}")
