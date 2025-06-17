from torch import nn
import torch
from typing import Union, Iterable

import torch.nn.functional as F

from torch.optim import AdamW, Adam, lr_scheduler, SGD
import pytorch_lightning as pl

import utils
import dataclasses

from training import TrainingContext

import adapt_modules as am
import wandb
import training
import paths


class BasicBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = am.Conv2d(
            in_channels, mid_channels,
            kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = am.BatchNorm2d(mid_channels)

        self.conv2 = am.Conv2d(mid_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = am.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                am.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                am.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        return F.relu(out)


def adapt_copy_from_prebuilt(src: nn.Module, dest: nn.Module, verbose=0):
    def find_instance_type(obj, *types):
        for t in types:
            if isinstance(obj, t):
                return t
        return None

    MODULE_TYPES = (
        torch.nn.Conv2d,
        torch.nn.Linear,
        torch.nn.BatchNorm2d,
    )

    dest_iter = iter(dest.named_modules())
    for src_name, src_module in src.named_modules():
        src_instance_type = find_instance_type(src_module, *MODULE_TYPES)
        if src_instance_type is None:
            continue

        while True:
            dest_name, dest_module = next(dest_iter)

            if dest_name.find('channel_attention') > 0:
                continue
            if dest_name.find('spatial_attention') > 0:
                continue

            if dest_name is None:
                return

            if not isinstance(dest_module, am.Module):
                if verbose >= 2:
                    print(
                        f"{src_name} not copied to {dest_name} because it is not an AdaptBaseNetWork")
                continue
            dest_module: am.Module
            if not isinstance(src_module, dest_module.base_type()):
                if verbose >= 2:
                    print(
                        f"{src_name} not copied to {dest_name} because it is not the same layer type")
                continue

            dest_name: str
            if verbose >= 1:
                print(f"copy from {src_name} to {dest_name}")
            dest_module.load_from_base(src_module)
            break


KNOWN_MODEL_PRETRAINED = {
    (10, (3, 3, 3), (16, 32, 64)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True),
    (10, (5, 5, 5), (16, 32, 64)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True),
    (10, (7, 7, 7), (16, 32, 64)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet44", pretrained=True),
    (10, (9, 9, 9), (16, 32, 64)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True),
    (100, (3, 3, 3), (16, 32, 64)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=True),
    (100, (5, 5, 5), (16, 32, 64)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet32", pretrained=True),
    (100, (7, 7, 7), (16, 32, 64)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet44", pretrained=True),
    (100, (9, 9, 9), (16, 32, 64)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=True),
}


class Resnet(nn.Module):
    def __init__(self, config: 'Config'):
        super().__init__()
        self.build_net(config)

    def build_net(self, config: 'Config'):
        self.levels = len(config.small_channels)

        self.conv1 = am.Conv2d(
            [3] * self.levels,
            config.small_channels,
            kernel_size=3, padding=1, bias=False)
        self.bn1 = am.BatchNorm2d(config.small_channels)

        self.layer1 = self._make_base_layer(
            config.small_channels, config.small_channels, config.num_blocks[0])
        self.layer2 = self._make_base_layer(
            config.small_channels, config.mid_channels, config.num_blocks[1], stride=2)
        self.layer3 = self._make_base_layer(
            config.mid_channels, config.large_channels, config.num_blocks[2], stride=2)

        self.fc = am.Linear(
            config.large_channels,
            [config.num_classes] * self.levels)

        self.set_level_use(self.levels - 1)

        if config.prebuilt:
            prebuild_config = (config.num_classes, config.num_blocks, (
                config.small_channels[-1], config.mid_channels[-1], config.large_channels[-1]))
            if prebuild_config not in KNOWN_MODEL_PRETRAINED:
                raise RuntimeError("prebuilt model not found")

            prebuilt = KNOWN_MODEL_PRETRAINED[prebuild_config]()
            adapt_copy_from_prebuilt(prebuilt, self)

    def _make_base_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(
            in_channels, out_channels, out_channels, stride))
        in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(in_channels, out_channels, out_channels))
        return nn.Sequential(*layers)

    def set_level_use(self, level):
        for _, module in self.named_modules():
            if isinstance(module, am.Module):
                module.set_level_use(level)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class FullTrainer(pl.LightningModule):
    def __init__(self, model: Resnet, upto):
        super().__init__()
        self.submodel = model
        self.upto = upto

    def forward(self, x):
        return self.submodel(x)

    def _step(self, batch, stage):
        x, y = batch
        total_loss = 0.0

        for i in range(self.submodel.levels):
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


@utils.fluent_setters
@dataclasses.dataclass
class Config(utils.SelfDescripting):
    num_blocks: Iterable[int] = (3, 3, 3)
    num_classes: int = 10
    small_channels: Iterable[int] = (8, 12, 16)
    mid_channels: Iterable[int] = (16, 24, 32)
    large_channels: Iterable[int] = (32, 48, 64)
    prebuilt: bool = True
    training_context: TrainingContext = None

    def run_training(self):
        torch.set_float32_matmul_precision('high')

        device = utils.get_device()
        model = Resnet(self).to(device)

        trainer = FullTrainer(model, model.levels - 1)

        with wandb.init(project="test adapt", name=self.get_description(), config=self.get_flat_dict(), dir=paths.LOG_PATH):
            training.finetune(trainer, self.training_context(device))
