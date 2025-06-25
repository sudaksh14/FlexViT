from torch import nn
import torch

from adapt_modules.module import Module
from adapt_modules.adapt_select import AdaptSelect


class LayerNorm(AdaptSelect):
    def __init__(self, channels, *args, **kwargs) -> None:
        self._args = args
        self._kwargs = kwargs
        self._channels = channels
        layers = [
            nn.LayerNorm(c, *args, **kwargs) for c in channels
        ]
        super().__init__(layers)

    @staticmethod
    def base_type() -> type[nn.LayerNorm]:
        return nn.LayerNorm

    def make_base_copy(self) -> nn.LayerNorm:
        m = nn.LayerNorm(
            self._channels[self.current_level()], *self._args, **self._kwargs)
        self.copy_to_base(m)
        return m
