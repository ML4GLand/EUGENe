# https://github.com/Lightning-AI/lightning/discussions/14782
import torch
import torchattacks
from torch import nn
from pytorch_lightning import LightningModule


class AdversarialModel(LightningModule):
    def __init__(
        self,
        model,
        attack=None,
        loss_fxn=nn.CrossEntropyLoss(),
        optim="AdamW",
        clean=False,
        lr=0.01,
    ):
        super().__init__()
        self.model = model
        self.loss_fxn = loss_fxn
        self.atk = attack
        self.clean = clean
        self.lr = lr
        if optim is None:
            self.optim = torch.optim.AdamW
        elif optim == "AdamW":
            self.optim = torch.optim.AdamW
        elif optim == "Adam":
            self.optim = torch.optim.Adam
        elif optim == "SGD":
            self.optim = torch.optim.SGD
        else:
            raise ValueError(f"Optim should be in '[AdamW, Adam, SGD]', not {optim}")

    def forward(self, x, clean=None):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        imgs, labels = batch
        if not self.clean:
            imgs = self.atk(imgs, labels)
        logits = self.model(imgs)
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        acc = (logits.argmax(dim=1)).eq(labels).sum().item() / len(imgs)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch

        clean_logits = self.model(imgs)
        clean_loss = self.loss_fn(clean_logits, labels)
        clean_acc = (clean_logits.argmax(dim=1)).eq(labels).sum().item() / len(imgs)
        self.log("clean_val_loss", clean_loss, prog_bar=True)
        self.log("clean_val_acc", clean_acc, prog_bar=True)
        return clean_loss, clean_acc

    def test_step(self, batch, batch_idx):
        imgs, labels = batch

        clean_logits = self.model(imgs)
        clean_loss = self.loss_fn(clean_logits, labels)
        clean_acc = (clean_logits.argmax(dim=1)).eq(labels).sum().item() / len(imgs)

        self.log("clean_test_loss", clean_loss, prog_bar=True)
        self.log("clean_test_acc", clean_acc, prog_bar=True)
        return clean_loss, clean_acc

    def configure_optimizers(self):
        optim = self.optim
        if issubclass(optim, torch.optim.SGD):
            if self.lr is not None:
                return optim(
                    self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4
                )
            else:
                return optim(self.model.parameters(), momentum=0.9, weight_decay=1e-4)
        elif issubclass(optim, (torch.optim.Adam, torch.optim.AdamW)):
            if self.lr is not None:
                return optim(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
            else:
                return optim(self.model.parameters(), weight_decay=1e-4)
        else:
            return self.optim
