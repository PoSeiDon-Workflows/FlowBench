""" MLP model for node classification """


import lightning as L
import torch
import torchmetrics
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from sklearn.metrics import accuracy_score
from torch_geometric.nn.models import MLP
from torchmetrics import ConfusionMatrix

from .base_model import BaseModel


class MLPClassifier(BaseModel):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(MLPClassifier, self).__init__(in_channels, out_channels, **kwargs)

        # self.in_channels = in_channels
        # self.out_channels = out_channels
        self.hidden_channels = kwargs.get('hidden_channels', 128)

        self.num_layers = kwargs.get('num_layers', 3)
        self.lr = kwargs.get('lr', 1e-4)
        self.dropout = kwargs.get('dropout', 0.1)
        self.mlp = MLP(in_channels=self.in_channels,
                       hidden_channels=self.hidden_channels,
                       out_channels=self.out_channels,
                       num_layers=self.num_layers,
                       dropout=self.dropout)
        # self.acc = torchmetrics.Accuracy(task='binary')
        # self.auroc = torchmetrics.AUROC(task='binary')
        # self.conf_mat = ConfusionMatrix(num_classes=out_channels)

    def forward(self, data):
        return self.mlp(data.x)

    def training_step(self, batch, batch_idx):
        z = self.forward(batch)
        loss_ = self.loss_fn(z, batch.y)
        self.log('train_loss', loss_)
        acc_ = self.acc(z.argmax(dim=1), batch.y)
        self.log('train_acc', acc_,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 batch_size=batch.y.size(0))
        # DEBUG: TORCH_USE_CUDA_DSA ?
        # auroc_ = self.auroc(z.argmax(dim=1), batch.y)
        # self.log('train_auroc', auroc_, on_epoch=True, prog_bar=True)
        return loss_

    def validation_step(self, batch, batch_idx):
        z = self.forward(batch)
        loss_ = self.loss_fn(z, batch.y)
        self.log('val_loss', loss_)
        acc_ = self.acc(z.argmax(dim=1), batch.y)
        self.log('val_acc', acc_,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 batch_size=batch.y.size(0))
        # auroc_ = self.auroc(z.argmax(dim=1), batch.y)
        # self.log('val_auroc', auroc_, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        z = self.forward(batch)
        loss_ = self.loss_fn(z, batch.y)
        self.log('test_loss', loss_)
        acc_ = self.acc(z.argmax(dim=1), batch.y)
        self.log('test_acc', acc_,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 batch_size=batch.y.size(0))

        # auroc_ = self.auroc(z.argmax(dim=1), batch.y)
        # self.log('test_auroc', auroc_, on_epoch=True, prog_bar=True)

    # def on_test_epoch_end(self, outputs):
    #     preds = torch.cat([x['preds'] for x in outputs]).cpu().numpy()
    #     y = torch.cat([x['y'] for x in outputs]).cpu().numpy()
    #     self.test_preds = preds
    #     self.test_y = y
    #     acc = accuracy_score(y_true=y, y_pred=preds)
    #     self.log('test_acc', acc, prog_bar=True)
