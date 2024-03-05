import os.path as osp

import lightning as L
import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader, RandomNodeLoader
# from torch_geometric.nn.models import GCN
from tqdm import tqdm

from flowbench.dataset import FlowDataset
from flowbench.supervised.gnn import GNN, GNN_v2
from flowbench.transforms import MinMaxNormalizeFeatures
from examples.utils import load_workflow
from torch_geometric.loader import NeighborLoader
torch.set_float32_matmul_precision("medium")

ds = load_workflow(workflow="casa_nowcast_full")
train_mask = ds[0].train_mask
val_mask = ds[0].val_mask
test_mask = ds[0].test_mask
# print(ds.y.bincount())
# exit()
# RandomNodeLoader()
train_loader = NeighborLoader(ds[0], num_neighbors=[-1, -1], batch_size=256, shuffle=True, input_nodes=train_mask)
val_loader = NeighborLoader(ds[0], num_neighbors=[-1, -1], batch_size=256, shuffle=False, input_nodes=val_mask)
test_loader = NeighborLoader(ds[0], num_neighbors=[-1, -1], batch_size=256, shuffle=False, input_nodes=test_mask)

# loader = DataLoader(ds, batch_size=256, shuffle=True)
model = GNN_v2(ds.num_node_features, ds.num_classes, num_layers=4, lr=1e-3)
print(model)
trainer = L.Trainer(max_epochs=10, logger=False, accelerator="gpu", devices=1)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.test(model, dataloaders=test_loader)

# ds = ds.to(torch.device('cuda'))
# loader = DataLoader(ds, batch_size=256, shuffle=True)
# model = GCN(ds.num_node_features, ds.num_classes, num_layers=1)
# model = model.to(torch.device('cuda'))
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# criterion = torch.nn.CrossEntropyLoss()

# pbar = tqdm(range(200))
# for epoch in pbar:
#     model.train()
#     total = 0
#     correct = 0
#     for data in loader:
#         data = data.to(torch.device('cuda'))
#         optimizer.zero_grad()
#         out = model(data.x, data.edge_index)
#         loss = criterion(out, data.y)
#         loss.backward()
#         optimizer.step()

#         # Calculate accuracy
#         _, predicted = torch.max(out.data, 1)
#         total += data.y.size(0)
#         correct += (predicted == data.y).sum().item()

#     accuracy = 100 * correct / total
#     pbar.set_postfix({"loss": f"{loss.item():.4f}", "accuracy": f"{accuracy:.2f}%"})
