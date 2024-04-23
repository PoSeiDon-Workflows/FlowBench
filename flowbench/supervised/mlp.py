""" MLP model for node classification

Author: PoSeiDon Team
License: MIT
"""
import lightning as L
import torch
import torchmetrics
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from sklearn.metrics import accuracy_score
from torch_geometric.nn.models import MLP
from torchmetrics import ConfusionMatrix

from .base_model import BaseModel


class MLPClassifier(BaseModel):
    r""" MLP classifier for node classification.

    Attributes:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        hidden_channels (int): Hidden channels.
        num_layers (int): Number of layers.
        lr (float): Learning rate.
        dropout (float): Dropout rate.
        mlp (torch_geometric.nn.models.MLP): MLP model.
        acc (torchmetrics.Accuracy): Accuracy metric.
        auroc (torchmetrics.AUROC): AUROC metric.
        conf_mat (torchmetrics.ConfusionMatrix): Confusion matrix.
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        r""" Initialize the MLP classifier.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
        """
        super(MLPClassifier, self).__init__(in_channels, out_channels, **kwargs)

        self.hidden_channels = kwargs.get('hidden_channels', 128)
        self.num_layers = kwargs.get('num_layers', 3)
        self.lr = kwargs.get('lr', 1e-4)
        self.dropout = kwargs.get('dropout', 0.1)
        self.mlp = MLP(in_channels=self.in_channels,
                       hidden_channels=self.hidden_channels,
                       out_channels=self.out_channels,
                       num_layers=self.num_layers,
                       dropout=self.dropout)

    def forward(self, data):
        r""" Forward pass of the MLP classifier.

        Args:
            data (torch_geometric.data.Data): Input data.
        """
        return self.mlp(data.x)

    def training_step(self, batch, batch_idx):
        r"""Here you compute and return the training loss and some additional metrics for e.g. the progress bar or logger.

        Args:
            batch (torch.utils.data.DataLoader): The output of your data iterable.
            batch_idx (int): The index of this batch.

        Return:
            dict: Dict with loss and accuracy score for the training step.
        """
        # forward pass
        z = self.forward(batch)
        # compute training loss
        loss_ = self.loss_fn(z, batch.y)
        # log training loss
        self.log('train_loss', loss_)
        # compute training accuracy
        acc_ = self.acc(z.argmax(dim=1), batch.y)
        # log training accuracy
        self.log('train_acc', acc_,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 batch_size=batch.y.size(0))
        return {'loss': loss_, 'acc': acc_}

    def validation_step(self, batch, batch_idx):
        r""" Operates on a single batch of data from the validation set. In this step you'd might generate examples or calculate anything of interest like accuracy.

        Args:
            batch (torch.utils.data.DataLoader): The output of your data iterable.
            batch_idx (int): The index of this batch.

        Return:
            dict: Dict with loss and accuracy score for the training step.
        """
        z = self.forward(batch)
        loss_ = self.loss_fn(z, batch.y)
        self.log('val_loss', loss_)
        acc_ = self.acc(z.argmax(dim=1), batch.y)
        self.log('val_acc', acc_,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 batch_size=batch.y.size(0))
        return {'loss': loss_, 'acc': acc_}

    def test_step(self, batch, batch_idx):
        r""" Operates on a single batch of data from the test set. In this step you'd normally generate examples or calculate anything of interest such as accuracy.

        Args:
            batch (torch.utils.data.DataLoader): The output of your data iterable.
            batch_idx (int): The index of this batch.

        Return:
            dict: Dict with loss and accuracy score for the training step.
        """
        z = self.forward(batch)
        loss_ = self.loss_fn(z, batch.y)
        self.log('test_loss', loss_)
        acc_ = self.acc(z.argmax(dim=1), batch.y)
        self.log('test_acc', acc_,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 batch_size=batch.y.size(0))
        return {'loss': loss_, 'acc': acc_}

    # NOTE: add ModelHooks to model
    # def on_test_epoch_end(self, outputs):
    #     preds = torch.cat([x['preds'] for x in outputs]).cpu().numpy()
    #     y = torch.cat([x['y'] for x in outputs]).cpu().numpy()
    #     self.test_preds = preds
    #     self.test_y = y
    #     acc = accuracy_score(y_true=y, y_pred=preds)
    #     self.log('test_acc', acc, prog_bar=True)
