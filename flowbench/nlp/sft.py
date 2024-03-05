""" Supervised fine-tuning of pretrained LLMs """

from sklearn.model_selection import PredefinedSplit
import torch
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pytorch_lightning import LightningModule


class SFT(LightningModule):
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        loss, logits = self.forward(input_ids, attention_mask, labels)
        preds = torch.argmax(logits, 1)
        acc = (preds == labels).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        loss, logits = self.forward(input_ids, attention_mask, labels)
        preds = torch.argmax(logits, 1)
        acc = (preds == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        loss, logits = self.forward(input_ids, attention_mask, labels)
        preds = torch.argmax(logits, 1)
        acc = (preds == labels).float().mean()
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-5)
