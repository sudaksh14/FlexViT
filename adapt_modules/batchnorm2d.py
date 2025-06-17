from torch import nn
import torch

from adapt_modules.module import Module
from adapt_modules.adapt_select import AdaptSelect


class BatchNorm2d(AdaptSelect):
    def __init__(self, channels, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._channels = channels
        layers = [
            nn.BatchNorm2d(c, *args, **kwargs) for c in channels
        ]
        super().__init__(layers)

    def base_type(self):
        return nn.BatchNorm2d

    def make_base_copy(self) -> nn.Linear:
        m = nn.BatchNorm2d(
            self._channels[self.current_level()], *self._args, **self._kwargs)
        self.copy_to_base(m)
        return m
