
import os.path as osp

import lightning as L
import torch
import torch_geometric.transforms as T
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.loader import DataLoader, RandomNodeLoader, NeighborLoader, ImbalancedSampler
# from torch_geometric.nn.models import GCN
from tqdm import tqdm

from flowbench.dataset import FlowDataset
from flowbench.supervised.mlp import MLPClassifier
from flowbench.transforms import MinMaxNormalizeFeatures
from examples.utils import load_workflow
torch.set_float32_matmul_precision("medium")

# ROOT = osp.join("/tmp", "data", "poseidon")
# WORKFLOW = "1000genome"
# pre_transform = T.Compose([MinMaxNormalizeFeatures(),
#                            T.ToUndirected()])

# ds = FlowDataset(root=ROOT,
#                  name=WORKFLOW,
#                  binary_labels=True,
#                  node_level=True,
#                  pre_transform=pre_transform,
#                  force_reprocess=True)
ds = load_workflow(workflow="casa_nowcast_small")
train_mask = ds[0].train_mask
val_mask = ds[0].val_mask
test_mask = ds[0].test_mask
# print(ds.y.bincount())
# exit()
# RandomNodeLoader()
train_loader = NeighborLoader(ds[0], num_neighbors=[-1, -1], batch_size=256, shuffle=True, input_nodes=train_mask)
val_loader = NeighborLoader(ds[0], num_neighbors=[-1, -1], batch_size=256, shuffle=False, input_nodes=val_mask)
test_loader = NeighborLoader(ds[0], num_neighbors=[-1, -1], batch_size=256, shuffle=False, input_nodes=test_mask)
# train_loader = RandomNodeLoader(ds[0], num_parts=128, shuffle=True, node_index=train_mask)
# val_loader = RandomNodeLoader(ds[0], num_parts=128, shuffle=False, node_index=val_mask)
model = MLPClassifier(ds.num_node_features, ds.num_classes, num_layers=4, lr=1e-3)
print(model)
trainer = L.Trainer(max_epochs=10, logger=False, accelerator="gpu", devices=1)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
# trainer.predict(model, dataloaders=test_loader)
trainer.test(model, dataloaders=test_loader)
