from torch import nn
import torch
from typing import Union, Iterable, List, cast

import torch.nn.functional as F

import utils
import dataclasses

from training import TrainingContext

import adapt_modules as am

from networks.adapt_model import AdaptModel

from networks.vgg import KNOWN_MODEL_PRETRAINED, LAYER_CONFIGS
from networks.config import ModelConfig


@utils.fluent_setters
@dataclasses.dataclass
class VGGConfig(ModelConfig):
    version: int = 11
    small_channels: int = (32, 48, 64)
    mid_channels: int = (64, 96, 128)
    large_channels: int = (128, 192, 256)
    max_channels: int = (256, 384, 512)
    num_classes: int = 10
    prebuilt: bool = True
    prebuilt_level: int = -1

    def make_model(self):
        return VGG(self)


class VGG(AdaptModel):
    def __init__(self, config: 'VGGConfig'):
        super().__init__()
        self.levels = len(config.small_channels)

        self.features = self.make_layers(
            LAYER_CONFIGS[config.version](
                config.small_channels,
                config.mid_channels,
                config.large_channels,
                config.max_channels))

        self.classifier = nn.Sequential(
            am.Linear(config.max_channels, config.max_channels),
            nn.ReLU(True),
            nn.Dropout(),
            am.Linear(config.max_channels, config.max_channels),
            nn.ReLU(True),
            nn.Dropout(),
            am.Linear(
                config.max_channels,
                [config.num_classes] * self.levels),
        )

        self.set_level_use(self.levels - 1)
        self.level = self.levels - 1

        if config.prebuilt:
            if config.prebuilt_level < 0:
                self.set_level_use(self.max_level() + 1 + config.prebuilt_level)
            else:
                self.set_level_use(config.prebuilt_level)
            prebuild_config = (
                config.num_classes, config.version,
                (config.small_channels[self.current_level()], config.mid_channels[self.current_level()], config.large_channels[self.current_level()], config.max_channels[self.current_level()]))
            if prebuild_config not in KNOWN_MODEL_PRETRAINED:
                raise RuntimeError("prebuilt model not found")
            prebuilt = KNOWN_MODEL_PRETRAINED[prebuild_config]()
            utils.flexible_model_copy(prebuilt, self)

    def make_layers(self, cfg):
        layers: List[nn.Module] = []
        in_channels = [3] * self.levels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = cast(int, v)
                conv2d = am.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, am.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def current_level(self) -> int:
        return self.level

    def max_level(self) -> int:
        return self.levels - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
