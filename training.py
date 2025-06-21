import torch
from torch.utils.data import DataLoader
import dataclasses

import pytorch_lightning as pl

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Timer
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from torch.optim import AdamW, Adam, lr_scheduler, SGD
import pytorch_lightning as pl

import paths
from networks.adapt_model import AdaptModel
import torch.nn.functional as F
import utils

import tempfile
from networks.adapt_model import AdaptModel
from networks.config import ModelConfig

import adapt_modules as am

import wandb
from typing import Callable


@utils.fluent_setters
@dataclasses.dataclass
class TrainingContext:
    loader_function: Callable[[], tuple[DataLoader, DataLoader, DataLoader]]

    patience: int = 5
    epochs: int = 10

    max_time: str = '00:23:00:00'

    def make_optimizer(self, model) -> torch.optim.Optimizer:
        raise NotImplemented()

    def make_scheduler(self, optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        raise NotImplemented()

    def wrap_model(self, model: pl.LightningModule) -> pl.LightningModule:
        def configure_optimizers():
            optimizer = self.make_optimizer(model)
            scheduler = self.make_scheduler(optimizer)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
        model.__dict__['configure_optimizers'] = configure_optimizers

    def unwrap_model(self, model: pl.LightningModule) -> pl.LightningModule:
        model.__dict__.pop('configure_optimizers')


@utils.fluent_setters
@dataclasses.dataclass
class AdaptiveTrainingContext(TrainingContext):
    incremental_training: bool = False


class BaseTrainer:
    def run_training(self, conf_description: str) -> None:
        raise NotImplemented()


class AdaptiveModelTrainer(pl.LightningModule, BaseTrainer):
    def __init__(self, model_config: ModelConfig, training_context: AdaptiveTrainingContext) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.training_context = training_context
        self.submodel = self.model_config.make_model().to(utils.get_device())
        self.upto = self.submodel.max_level()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.submodel(x)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
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

    def training_step(self, b, _) -> torch.Tensor:
        return self._step(b, "train")

    def validation_step(self, b, _) -> torch.Tensor:
        return self._step(b, "val")

    def test_step(self, b, _) -> torch.Tensor:
        return self._step(b, "test")

    def run_training(self, conf_description: str) -> None:
        torch.set_float32_matmul_precision('high')

        model = self.submodel
        trainer = self

        if self.training_context.incremental_training:
            self.upto = 0

        with wandb.init(project="a", name=conf_description, config=self.model_config.get_flat_dict(), dir=paths.LOG_PATH):
            if self.training_context.incremental_training:
                for i in range(model.max_level() + 1):
                    trainer = finetune(trainer, self.training_context)
                    trainer.upto += 1
            else:
                trainer = finetune(trainer, self.training_context)

        utils.save_model(
            trainer, self.model_config.get_filename_safe_description())

    def configure_optimizers(self) -> None:
        pass


def make_zero_grad_optimizer(optimizer, model: AdaptModel, freeze_level, *args, **kwargs):
    class zerograd(optimizer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def step(self, closure=None):
            for module in model.modules():
                if not isinstance(module, am.Module):
                    continue
                module.zero_out_gradients(freeze_level)
            return super().step(closure)

    return zerograd(*args, **kwargs)


@utils.fluent_setters
@dataclasses.dataclass
class ZeroOutTrainingContext(TrainingContext):
    zero_out_level: int = -1


class ZeroOutTrainer(pl.LightningModule, BaseTrainer):
    def __init__(self, model_config: ModelConfig, training_context: ZeroOutTrainingContext) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.training_context = training_context
        self.submodel: AdaptModel = utils.load_model(
            self.model_config.get_filename_safe_description()).submodel
        self.level_use = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.submodel(x)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
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

            if self.level_use == i:
                total_loss = loss

        self.log(f"{stage}_loss", total_loss, prog_bar=False)
        return total_loss

    def training_step(self, b, _) -> torch.Tensor:
        return self._step(b, "train")

    def validation_step(self, b, _) -> torch.Tensor:
        return self._step(b, "val")

    def test_step(self, b, _) -> torch.Tensor:
        return self._step(b, "test")

    def run_training(self, conf_description: str) -> None:
        torch.set_float32_matmul_precision('high')

        model = self.submodel
        trainer = self

        with wandb.init(project="a", name=conf_description, config=self.model_config.get_flat_dict(), dir=paths.LOG_PATH):
            for i in range(model.max_level() + 1):
                model.set_level_use(i)
                self.training_context.zero_out_level = i - 1
                trainer = finetune(trainer, self.training_context)

        utils.save_model(
            trainer, self.model_config.get_filename_safe_description())


def finetune(model: pl.LightningModule, config: TrainingContext) -> pl.LightningModule:
    logger = WandbLogger(log_model=False, dir=paths.LOG_PATH)
    config.wrap_model(model)

    with tempfile.TemporaryDirectory() as tdir:
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=config.patience, mode='min', verbose=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=tdir,
            filename='best-model',
            monitor='val_loss',
            mode='min',
            save_top_k=1
        )

        timer = Timer(config.max_time)

        trainer = pl.Trainer(
            max_epochs=config.epochs,
            logger=logger,
            callbacks=[early_stopping, checkpoint_callback, timer],
            log_every_n_steps=10,
            enable_checkpointing=True,
            accelerator="gpu",
            devices="auto"
        )

        train_loader, val_loader, test_loader = config.loader_function()
        trainer.fit(model, train_loader, val_loader)
        model = type(model).load_from_checkpoint(
            checkpoint_callback.best_model_path)
        config.wrap_model(model)
        trainer.test(model, dataloaders=test_loader)

    config.unwrap_model(model)
    return model
