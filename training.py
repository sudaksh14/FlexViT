from typing import Callable, Optional
import dataclasses
import tempfile
import datetime
import logging

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch

from networks.config import ModelConfig, FlexModelConfig
import config.paths as paths
import config.wandb
import utils


@dataclasses.dataclass
class TrainingContext(utils.SelfDescripting):
    loader_function: Callable[[], tuple[DataLoader, DataLoader, DataLoader]]
    patience: int = 5
    epochs: int = 10
    label_smoothing: float = 0.0
    gradient_clip_val: Optional[float] = None

    wandb_project_name: str = config.wandb.WANDB_PROJECT_NAME

    unittest_mode: bool = False

    def make_optimizer(self, model) -> torch.optim.Optimizer:
        raise NotImplementedError()

    def make_scheduler(self, optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        raise NotImplementedError()


class BaseTrainer:
    def get_model(self) -> nn.Module:
        raise NotImplementedError()

    def run_training(self, conf_description: str) -> None:
        raise NotImplementedError()


@dataclasses.dataclass
class TrainerBuilder:
    training_method: type[BaseTrainer]
    model_config: ModelConfig
    training_context: TrainingContext

    def __init__(self, training_method: type[BaseTrainer], model_config: ModelConfig, training_context: TrainingContext):
        self.training_method = training_method
        self.model_config = model_config
        self.training_context = training_context

    def build(self):
        return self.training_method(
            self.model_config, self.training_context)

    def run_training(self, conf: str):
        return self.build().run_training(conf)

    def __call__(self, conf: str):
        return self.run_training(conf)


@dataclasses.dataclass
class FlexTrainingContext(TrainingContext):
    load_from: Optional[ModelConfig] = None
    distill: bool = False


class FlexModelTrainer(pl.LightningModule, BaseTrainer):
    def __init__(self, model_config: ModelConfig, training_context: FlexTrainingContext) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.training_context = training_context
        self.submodel = self.model_config.make_model()
        self.distill_net = None
        self.automatic_optimization = False
        self.Mixup = utils.mixup_fn


    def get_model(self) -> nn.Module:
        return self.submodel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.submodel(x)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        x, y = batch

        if stage == "train":
            x,y = self.Mixup(x, y)

        if self.distill_net is not None:
            self.distill_net.eval()
            for p in self.distill_net.parameters():
                p.requires_grad_(False)
            y_loss = self.distill_net(x)
        else:
            y_loss = y

        total_loss = 0.0

        for i in range(self.submodel.max_level() + 1):
            self.submodel.set_level_use(i)
            logits = self(x)
            loss = F.cross_entropy(
                logits, y_loss, label_smoothing=self.training_context.label_smoothing)
                        
            # Handle soft labels for Mixup/CutMix
            if y.ndim == 2:
                acc = (logits.argmax(1) == y.argmax(1)).float().mean()
            else:
                acc = (logits.argmax(1) == y).float().mean()

            self.log(f"{stage}_level{i}_loss", loss,
                     prog_bar=False, sync_dist=True)
            self.log(f"{stage}_level{i}_acc",  acc,
                     prog_bar=(stage != 'train'), sync_dist=True)
            if stage == "train":
                self.manual_backward(loss)
            total_loss += loss.clone().detach()

        self.log(f"{stage}_loss", total_loss, prog_bar=(
            stage != 'train'), sync_dist=True)
        if stage == "train":
            opt = self.optimizers()
            opt.zero_grad()
            opt.step()

    def training_step(self, b, _) -> torch.Tensor:
        return self._step(b, "train")

    def validation_step(self, b, _) -> torch.Tensor:
        return self._step(b, "val")

    def test_step(self, b, _) -> torch.Tensor:
        return self._step(b, "test")

    def handle_load_from(self):
        if self.training_context.load_from is not None:
            lmodel = utils.load_model(self.training_context.load_from)
            utils.flexible_model_copy(lmodel, self.submodel)

    def handle_distill(self):
        if self.training_context.distill:
            distill_config = self.model_config.no_prebuilt()
            if isinstance(distill_config, FlexModelConfig):
                distill_config = distill_config.create_base_config(
                    self.submodel.max_level())
            self.distill_net = distill_config.make_model()
            utils.flexible_model_copy(self.submodel, self.distill_net)

    def train_loop(self, trainer, conf_description):
        trainer = finetune(
            trainer, self.training_context,
            conf_description, self.model_config)

    def run_training(self, conf_description: str) -> None:
        torch.set_float32_matmul_precision('high')

        self.handle_load_from()
        self.handle_distill()
        self.train_loop(self, conf_description)

        # utils.save_model(self.model_config, self.submodel)
        # utils.save_statedict("FlexViT_10Levels", self.submodel)
        utils.save_statedict("FlexViT_5Levels", self.submodel)

    def configure_optimizers(self):
        optimizer = self.training_context.make_optimizer(self.submodel)
        scheduler = self.training_context.make_scheduler(optimizer)
        return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler,
            'monitor': 'val_loss',
            'interval': 'epoch',  # Update the scheduler every epoch
            'frequency': 1        # Frequency of updates
        }
    }



class SimpleTrainer(pl.LightningModule, BaseTrainer):
    def __init__(self, model_config: ModelConfig, training_context: TrainingContext) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.training_context = training_context
        self.submodel = self.model_config.make_model()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.submodel(x)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=False, sync_dist=True)
        self.log(f"{stage}_acc",  acc,
                 prog_bar=(stage != 'train'), sync_dist=True)
        return loss

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

        trainer = finetune(
            trainer, self.training_context,
            conf_description, self.model_config)

        utils.save_model(self.model_config, trainer.submodel)

    def configure_optimizers(self):
        optimizer = self.training_context.make_optimizer(self.submodel)
        scheduler = self.training_context.make_scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}


logger = None


def finetune(model: pl.LightningModule, config: TrainingContext, conf_description, model_config) -> pl.LightningModule:
    global logger

    if config.unittest_mode:
        logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
        logging.getLogger(
            'lightning_fabric.utilities.distributed').setLevel(logging.ERROR)

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

        callbacks = [early_stopping, checkpoint_callback]

        kwargs = dict()
        if config.wandb_project_name is not None:
            if logger is None:
                logger = WandbLogger(
                    project=config.wandb_project_name,
                    name="FlexViT-5level",
                    config=model_config.get_flat_dict(),
                    save_dir=paths.LOG_PATH,
                    dir=paths.LOG_PATH,
                    log_model=False)
            kwargs['logger'] = logger
        else:
            kwargs['logger'] = False

        if config.unittest_mode:
            kwargs['enable_progress_bar'] = False
            kwargs['enable_model_summary'] = False

        ddp = DDPStrategy(process_group_backend='nccl', find_unused_parameters=True)
        trainer = pl.Trainer(
            **kwargs,
            max_epochs=config.epochs,
            callbacks=callbacks,
            log_every_n_steps=10,
            enable_checkpointing=True,
            accelerator="gpu",
            devices="auto",
            num_nodes=utils.get_num_nodes(),
            strategy=ddp,
            precision='bf16-mixed'
        )

        train_loader, val_loader, test_loader = config.loader_function()
        trainer.fit(model, train_loader, val_loader)

        if utils.get_num_nodes() > 1:
            if trainer.is_global_zero:
                model = type(model).load_from_checkpoint(
                        checkpoint_callback.best_model_path)
            
        else:
            model = type(model).load_from_checkpoint(
                    checkpoint_callback.best_model_path)
        trainer.test(model, dataloaders=test_loader, verbose=False)

    return model
