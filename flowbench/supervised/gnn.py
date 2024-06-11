import lightning as L
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GCN
import torchmetrics
import torch.nn.functional as F
from torch.nn import Linear, ReLU, ModuleList, Sequential


class GNN(L.LightningModule):
    r""" GNN model for node classification """

    def __init__(self, num_features, num_classes, **kwargs):
        super(GNN, self).__init__()
        self.hidden_dim = kwargs.get('hidden_dim', 128)
        self.num_layers = kwargs.get('num_layers', 3)
        self.lr = kwargs.get('lr', 1e-4)
        self.dropout = kwargs.get('dropout', 0.5)

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_features, self.hidden_dim))
        for _ in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))

        self.seqs = torch.nn.ModuleList()
        for _ in range(self.num_layers - 1):
            self.seqs.append(Linear(self.hidden_dim, self.hidden_dim))

        self.seqs.append(Linear(self.hidden_dim, num_classes))

        # self.mlp = torch.nn.Sequential(
        #     torch.nn.Linear(self.hidden_dim, self.hidden_dim),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(self.hidden_dim, num_classes)
        # )

        self.acc = torchmetrics.Accuracy(task='binary')
        self.acu = torchmetrics.AUROC(task='binary')

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

        for seq in self.seqs:
            x = seq(x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x = self.forward(batch)
        loss = torch.nn.functional.cross_entropy(x, batch.y)
        self.log('train_loss', loss)
        acc = self.acc(x.argmax(dim=1), batch.y)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)
        auc = self.acu(x.argmax(dim=1), batch.y)
        self.log('train_auc', auc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.forward(batch)
        loss = torch.nn.functional.cross_entropy(x, batch.y)
        self.log('val_loss', loss)
        acc = self.acc(x.argmax(dim=1), batch.y)
        self.log('val_acc', acc, on_epoch=True)
        auc = self.acu(x.argmax(dim=1), batch.y)
        self.log('val_auc', auc, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = self.forward(batch)
        loss = torch.nn.functional.cross_entropy(x, batch.y)
        self.log('test_loss', loss)
        acc = self.acc(x.argmax(dim=1), batch.y)
        self.log('test_acc', acc, on_epoch=True)
        auc = self.acu(x.argmax(dim=1), batch.y)
        self.log('test_auc', auc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class GNN_v2(L.LightningModule):
    r""" GNN model for node classification
    NOTE: the version used in WORKS'22 paper
    """

    def __init__(self, num_features, num_classes, **kwargs):
        super(GNN_v2, self).__init__()
        self.hidden_dim = kwargs.get('hidden_dim', 128)
        self.num_layers = kwargs.get('num_layers', 3)
        self.lr = kwargs.get('lr', 1e-4)
        self.dropout = kwargs.get('dropout', 0.5)

        # add the ability to add one or more conv layers
        conv_blocks = [
            GCNConv(num_features, self.hidden_dim),
            ReLU(),
        ]

        # ability to  add one or more conv blocks
        for _ in range(self.num_layers - 1):
            conv_blocks += [
                GCNConv(self.hidden_dim, self.hidden_dim),
                ReLU(),
                GCNConv(self.hidden_dim, self.hidden_dim),
                ReLU(),
            ]

        # group all the conv layers
        self.conv_layers = ModuleList(conv_blocks)

        # add the linear layers for flattening the output from MPNN
        self.flatten = Sequential(
            Linear(self.hidden_dim, self.hidden_dim),
            ReLU(),
            Linear(self.hidden_dim, num_classes))

        self.acc = torchmetrics.Accuracy(task='binary')
        self.auroc = torchmetrics.AUROC(task='binary')

    def forward(self, data):
        # process the layers
        x, edge_index = data.x, data.edge_index
        for idx, layer in enumerate(self.conv_layers):
            if isinstance(layer, GCNConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # pass the output to the linear output layer
        out = self.flatten(x)

        # return the output
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x = self.forward(batch)
        loss = torch.nn.functional.cross_entropy(x, batch.y)
        # self.log('train_loss', loss)
        # acc = self.acc(x.argmax(dim=1), batch.y)
        # self.log('train_acc', acc, on_epoch=False, prog_bar=False, on_step=False)
        # auc = self.auroc(x.argmax(dim=1), batch.y)
        # self.log('train_auc', auc, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.forward(batch)
        loss = torch.nn.functional.cross_entropy(x, batch.y)
        self.log('val_loss', loss)
        acc = self.acc(x.argmax(dim=1), batch.y)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, on_step=False)
        auc = self.auroc(x.argmax(dim=1), batch.y)
        self.log('val_auc', auc, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = self.forward(batch)
        loss = torch.nn.functional.cross_entropy(x, batch.y)
        self.log('test_loss', loss)
        acc = self.acc(x.argmax(dim=1), batch.y)
        self.log('test_acc', acc, on_epoch=True)
        auc = self.auroc(x.argmax(dim=1), batch.y)
        self.log('test_auc', auc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class PyG_GNN(L.LightningModule):
    r""" GNN model for node classification
    NOTE: used PyG's GCN model
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = kwargs.get('num_layers', 3)
        self.hidden_channels = kwargs.get('hidden_channels', 128)
        self.gcn = GCN(in_channels=self.in_channels,
                       hidden_channels=self.hidden_channels,
                       out_channels=self.out_channels,
                       num_layers=self.num_layers)

    def forward(self, data):
        return self.gcn(data.x, data.edge_index)
