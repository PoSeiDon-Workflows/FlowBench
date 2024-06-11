
import os.path as osp

import lightning as L
import torch
import torch_geometric.transforms as T
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.loader import (DataLoader, ImbalancedSampler,
                                    NeighborLoader, RandomNodeLoader)
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
# from torch_geometric.nn.models import GCN
from tqdm import tqdm

from examples.utils import load_workflow
from flowbench.dataset import FlowDataset
from flowbench.supervised.mlp import MLPClassifier
from flowbench.transforms import MinMaxNormalizeFeatures
import time

torch.set_float32_matmul_precision("medium")

ds = load_workflow(workflow="somospie")
train_mask = ds[0].train_mask
val_mask = ds[0].val_mask
test_mask = ds[0].test_mask
# print(ds.y.bincount())
# exit()
# RandomNodeLoader()
train_loader = NeighborLoader(ds[0], num_neighbors=[-1, -1], batch_size=2048, shuffle=True, input_nodes=train_mask)
val_loader = NeighborLoader(ds[0], num_neighbors=[-1, -1], batch_size=2048, shuffle=False, input_nodes=val_mask)
test_loader = NeighborLoader(ds[0], num_neighbors=[-1, -1], batch_size=2048, shuffle=False, input_nodes=test_mask)
# train_loader = RandomNodeLoader(ds[0], num_parts=128, shuffle=True, node_index=train_mask)
# val_loader = RandomNodeLoader(ds[0], num_parts=128, shuffle=False, node_index=val_mask)
model = MLPClassifier(ds.num_node_features, ds.num_classes, num_layers=4, lr=5e-4)
print(model)
trainer = L.Trainer(
    max_epochs=200,
    logger=False,
    accelerator="gpu",
    devices=1,
    # callbacks=[
    #     EarlyStopping(
    #         monitor="val_loss",
    #         mode="min")]
)
tic = time.time()
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
toc = time.time()
print(f"Training time: {toc - tic}")
# trainer.predict(model, dataloaders=test_loader)
trainer.test(model, dataloaders=test_loader)
