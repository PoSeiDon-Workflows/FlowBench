""" Base model for supervised learning.

Author: PoSeiDon Team
License: MIT
"""
import lightning as L
import torch.nn.functional as F
import torchmetrics
from torch.optim import Adam


class BaseModel(L.LightningModule):
    r""" Base model for supervised learning.

    Attributes:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        lr (float): Learning rate.
        acc (torchmetrics.Accuracy): Accuracy metric.
        auroc (torchmetrics.AUROC): AUROC metric.
        loss_fn (torch.nn.CrossEntropyLoss): Loss function.
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        r""" Initialize the base model.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lr = kwargs.get('lr', 1e-4)

        super(BaseModel, self).__init__()

        if self.out_channels == 2:
            self.acc = torchmetrics.Accuracy(task='binary')
            self.auroc = torchmetrics.AUROC(task='binary')
            self.conf_mat = torchmetrics.ConfusionMatrix(task='binary', num_classes=2)
        else:
            self.acc = torchmetrics.Accuracy(task='multiclass')
            self.auroc = torchmetrics.AUROC(task='multiclass')
            self.conf_mat = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=out_channels)
        self.loss_fn = F.cross_entropy

    def reset_parameters(self):
        r""" Reset the parameters of the model. """
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def configure_optimizers(self):
        r""" Configure the optimizer.

        Returns:
            torch.optim.Optimizer: Adam optimizer.
        """
        return Adam(self.parameters(), lr=self.lr)
