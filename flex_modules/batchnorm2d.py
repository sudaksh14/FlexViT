from torch import nn
import torch

from flex_modules.module import Module
from flex_modules.flexselect import AdaptSelect


class BatchNorm2d(AdaptSelect):
    def __init__(self, channels, *args, **kwargs) -> None:
        self._args = args
        self._kwargs = kwargs
        self._channels = channels
        layers = [
            nn.BatchNorm2d(c, *args, **kwargs) for c in channels
        ]
        super().__init__(layers)

    @staticmethod
    def base_type() -> type[nn.BatchNorm2d]:
        return nn.BatchNorm2d

    @torch.no_grad()
    def make_base_copy(self) -> nn.BatchNorm2d:
        m = nn.BatchNorm2d(
            self._channels[self.current_level()], *self._args, **self._kwargs)
        self.copy_to_base(m)
        return m
