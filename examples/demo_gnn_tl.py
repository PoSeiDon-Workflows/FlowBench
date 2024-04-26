""" Transfer learning with GNN on Poseidon datasets 

1. train with 1000genome, montage, predict_future_sales
2. test with three workflows
3. test with casa_wind_full (transfer learning) - zero-shot
4. retrain with additional casa_wind_full
5. test with casa_wind_full 
"""
import os.path as osp

import lightning as L
import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader, NeighborLoader, RandomNodeLoader
# from torch_geometric.nn.models import GCN
from tqdm import tqdm

from examples.utils import load_workflow
from flowbench.dataset import MergeFlowDataset, FlowDataset
from flowbench.dataset_bak import FlowBench
from flowbench.supervised.gnn import GNN, GNN_v2
from flowbench.transforms import MinMaxNormalizeFeatures

torch.set_float32_matmul_precision("medium")

ROOT = osp.join("/tmp", "data", "poseidon")
pre_transform = T.Compose([MinMaxNormalizeFeatures(),
                           T.ToUndirected(),
                           T.RandomNodeSplit(split="train_rest",
                                             num_val=0.1,
                                             num_test=0.2)])
ds = MergeFlowDataset(root=ROOT,
                      name=['1000genome', 'montage', 'predict_future_sales'],
                      binary_labels=True,
                      node_level=True,
                      pre_transform=pre_transform,
                      force_reprocess=True)

fl_ds = FlowDataset(root=ROOT,
                    name="casa_wind_full",
                    binary_labels=True,
                    node_level=True,
                    pre_transform=pre_transform,
                    force_reprocess=True)
train_mask = ds[0].train_mask
val_mask = ds[0].val_mask
test_mask = ds[0].test_mask

fl_train_mask = fl_ds[0].train_mask
fl_val_mask = fl_ds[0].val_mask
fl_test_mask = fl_ds[0].test_mask

train_loader = NeighborLoader(ds[0], num_neighbors=[-1, -1], batch_size=256, shuffle=True, input_nodes=train_mask)
val_loader = NeighborLoader(ds[0], num_neighbors=[-1, -1], batch_size=256, shuffle=False, input_nodes=val_mask)
test_loader = NeighborLoader(ds[0], num_neighbors=[-1, -1], batch_size=256, shuffle=False, input_nodes=test_mask)


model = GNN_v2(ds.num_node_features, ds.num_classes, num_layers=4, lr=1e-3)
print(model)
trainer = L.Trainer(max_epochs=200, logger=False, accelerator="gpu", devices=1)
# fit model with three workflows
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# test model with three workflows
trainer.test(model, dataloaders=test_loader)

fl_train_loader = NeighborLoader(fl_ds[0], num_neighbors=[-1, -1], batch_size=256,
                                 shuffle=True, input_nodes=fl_train_mask)
fl_val_loader = NeighborLoader(fl_ds[0], num_neighbors=[-1, -1], batch_size=256,
                               shuffle=False, input_nodes=fl_val_mask)
fl_test_loader = NeighborLoader(fl_ds[0], num_neighbors=[-1, -1], batch_size=256,
                                shuffle=False, input_nodes=fl_test_mask)

# test model with fl_ds casa_wind_full
trainer.test(model, dataloaders=fl_test_loader)
# retrain the model with additional workflow
trainer.fit(model, train_dataloaders=fl_train_loader, val_dataloaders=fl_val_loader)
# test model with fl_ds casa_wind_full after train with additional workflow
trainer.test(model, dataloaders=fl_test_loader)
