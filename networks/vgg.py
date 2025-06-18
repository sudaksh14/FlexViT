import torch
import torch.nn as nn


from typing import Union, List, Dict, cast

import utils
import dataclasses

from training import TrainingContext
from training import SimpleTrainer
import training
import wandb
import paths

# basic implementation from github.com/chenyaofo/pytorch-cifar-models


@utils.fluent_setters
@dataclasses.dataclass
class VGGConfig(utils.SelfDescripting):
    version: int = 11
    small_channels: int = 64
    mid_channels: int = 128
    large_channels: int = 256
    max_channels: int = 512
    num_classes: int = 10
    prebuilt: bool = True
    training_context: TrainingContext = None

    def run_training(self, conf_description: str):
        torch.set_float32_matmul_precision('high')

        device = utils.get_device()
        model = VGG(self).to(device)

        trainer = SimpleTrainer(model)

        with wandb.init(project="test adapt", name=conf_description, config=self.get_flat_dict(), dir=paths.LOG_PATH):
            training.finetune(trainer, self.training_context(device))


LAYER_CONFIGS = {
    11: lambda a, b, c, d: [a, 'M', b, 'M', c, c, 'M', d, d, 'M', d, d, 'M'],
    13: lambda a, b, c, d: [a, a, 'M', b, b, 'M', c, c, 'M', d, d, 'M', d, d, 'M'],
    16: lambda a, b, c, d: [a, a, 'M', b, b, 'M', c, c, c, 'M', d, d, d, 'M', d, d, d, 'M'],
    19: lambda a, b, c, d: [a, a, 'M', b, b, 'M', c, c, c, c, 'M', d, d, d, d, 'M', d, d, d, d, 'M'],
}

KNOWN_MODEL_PRETRAINED = {
    (10, 11, (64, 128, 256, 512)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg11_bn", pretrained=True),
    (10, 13, (64, 128, 256, 512)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg13_bn", pretrained=True),
    (10, 16, (64, 128, 256, 512)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True),
    (10, 19, (64, 128, 256, 512)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg19_bn", pretrained=True),
    (100, 11, (64, 128, 256, 512)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg11_bn", pretrained=True),
    (100, 13, (64, 128, 256, 512)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg13_bn", pretrained=True),
    (100, 16, (64, 128, 256, 512)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg16_bn", pretrained=True),
    (100, 19, (64, 128, 256, 512)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg19_bn", pretrained=True),
}


class VGG(nn.Module):
    def __init__(self, config: VGGConfig):
        super().__init__()
        self.features = self.make_layers(LAYER_CONFIGS[config.version](
            config.small_channels, config.mid_channels, config.large_channels, config.max_channels))

        self.classifier = nn.Sequential(
            nn.Linear(config.max_channels, config.max_channels),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(config.max_channels, config.max_channels),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(config.max_channels, config.num_classes),
        )

        if config.prebuilt:
            prebuild_config = (
                config.num_classes, config.version,
                (config.small_channels, config.mid_channels, config.large_channels, config.max_channels))
            if prebuild_config not in KNOWN_MODEL_PRETRAINED:
                raise RuntimeError("prebuilt model not found")
            prebuilt = KNOWN_MODEL_PRETRAINED[prebuild_config]()
            utils.flexible_model_copy(prebuilt, self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def make_layers(self, cfg):
        layers: List[nn.Module] = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = cast(int, v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
