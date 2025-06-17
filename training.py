import dataclasses

from torch.utils.data import DataLoader
import torch

import pytorch_lightning as pl
import copy


from pytorch_lightning.callbacks import EarlyStopping

import pytorch_lightning as pl

import wandb
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import paths


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
