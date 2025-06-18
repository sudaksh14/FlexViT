from torch import nn
import torch

from adapt_modules.module import Module
from adapt_modules.adapt_select import AdaptSelect


class Linear(AdaptSelect):
    def __init__(self, in_sizes, out_sizes, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._in_sizes = in_sizes
        self._out_sizes = out_sizes
        layers = [
            nn.Linear(ins, outs, *args, **kwargs) for ins, outs in zip(in_sizes, out_sizes)
        ]
        super().__init__(layers)

    @staticmethod
    def base_type():
        return nn.Linear

    def make_base_copy(self) -> nn.Linear:
        m = nn.Linear(
            self._in_sizes[self.current_level()], self._out_sizes[self.current_level()], *self._args, **self._kwargs)
        self.copy_to_base(m)
        return m

    def export_level_delta(self):
        return ((self.layers[self.level].weight.data, self.layers[self.level].bias.data), (self.layers[self.level].weight.data, self.layers[self.level].bias.data))

    @staticmethod
    def apply_level_delta_down(model: nn.BatchNorm2d, level_delta):
        model.weight.data = level_delta[0][:]
        model.bias.data = level_delta[1][:]

    @staticmethod
    def apply_level_delta_up(model: nn.Module, level_delta):
        model.weight.data = level_delta[0][:]
        model.bias.data = level_delta[1][:]
