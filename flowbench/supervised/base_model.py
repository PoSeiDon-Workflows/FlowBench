""" Base model for supervised learning. """
import lightning as L
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torchmetrics
import torch.nn.functional as F
from torch.optim import Adam


class BaseModel(L.LightningModule):

    def __init__(self, in_channels, out_channels, **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lr = kwargs.get('lr', 1e-4)

        super(BaseModel, self).__init__()

        self.acc = torchmetrics.Accuracy(task='binary')
        self.auroc = torchmetrics.AUROC(task='binary')
        self.loss_fn = F.cross_entropy

    def reset_parameters(self):
        r""" Reset the parameters of the model. """
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
