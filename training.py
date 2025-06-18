import torch
from torch.utils.data import DataLoader
import dataclasses

import pytorch_lightning as pl

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from torch.optim import AdamW, Adam, lr_scheduler, SGD
import pytorch_lightning as pl

import paths
from networks.adapt_model import AdaptModel
import torch.nn.functional as F

@dataclasses.dataclass
class TrainingContext:
    device: torch.device

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader

    patience: int = 5
    epochs: int = 10

    def make_optimizer(self, model):
        raise NotImplemented()

    def make_scheduler(self, optimizer):
        raise NotImplemented()

    def wrap_model(self, model):
        def configure_optimizers():
            optimizer = self.make_optimizer(model)
            scheduler = self.make_scheduler(optimizer)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
        model.__dict__['configure_optimizers'] = configure_optimizers

    def unwrap_model(self, model: torch.nn.Module):
        model.__dict__.pop('configure_optimizers')


class FullTrainer(pl.LightningModule):
    def __init__(self, model: AdaptModel, upto):
        super().__init__()
        self.submodel = model
        self.upto = upto

    def forward(self, x):
        return self.submodel(x)

    def _step(self, batch, stage):
        x, y = batch
        total_loss = 0.0

        for i in range(self.submodel.max_level() + 1):
            self.submodel.set_level_use(i)
            logits = self(x)
            loss = F.cross_entropy(logits, y)
            acc = (logits.argmax(1) == y).float().mean()
            self.log(f"{stage}_level{i}_loss", loss, prog_bar=False)
            self.log(f"{stage}_level{i}_acc",  acc,
                     prog_bar=(stage != 'train'))
            if self.upto >= i:
                total_loss += loss

        self.log(f"{stage}_loss", total_loss, prog_bar=False)
        return total_loss

    def training_step(self, b, _): return self._step(b, "train")
    def validation_step(self, b, _): return self._step(b, "val")
    def test_step(self, b, _): return self._step(b, "test")

    def configure_optimizers(self):
        opt = AdamW(self.parameters(),
                    lr=self.hparams.config.learning_rate, weight_decay=1e-4)
        sched = lr_scheduler.ReduceLROnPlateau(
            opt, "min", factor=0.5, patience=3)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}


def finetune(model, config: TrainingContext):
    logger = WandbLogger(log_model=False, dir=paths.LOG_PATH)
    config.wrap_model(model)

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=config.patience, mode='min', verbose=True)

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        logger=logger,
        callbacks=[early_stopping],
        log_every_n_steps=10,
        enable_checkpointing=False,
        accelerator="gpu",
        devices="auto"
    )

    trainer.fit(model, config.train_loader, config.val_loader)
    trainer.test(model, dataloaders=config.test_loader)

    config.unwrap_model(model)
    return model
