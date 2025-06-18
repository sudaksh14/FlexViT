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

    def base_type(self):
        return nn.Linear

    def make_base_copy(self) -> nn.Linear:
        m = nn.Linear(
            self._in_sizes[self.current_level()], self._out_sizes[self.current_level()], *self._args, **self._kwargs)
        self.copy_to_base(m)
        return m
