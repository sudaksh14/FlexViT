from torch import nn
import torch

from adapt_modules.module import Module
import torch.nn.functional as F

from typing import Iterable
import copy


class Linear(Module):
    def __init__(self, in_sizes: Iterable[int], out_sizes: Iterable[int], *args, **kwargs):
        super().__init__()
        self.in_sizes = in_sizes
        self.out_sizes = out_sizes

        self._args = args
        self._kwargs = kwargs

        self.max_in_size = self.in_sizes[-1]
        self.max_out_size = self.out_sizes[-1]

        self.level = self.max_level()
        self.linear = nn.Linear(
            self.max_in_size, self.max_out_size, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, self.max_in_size - self.in_sizes[self.level]))
        x = self.linear(x)
        x = x[..., :self.out_sizes[self.level]]
        return x

    def set_level_use(self, level: int) -> None:
        assert (level >= 0)
        assert (level <= self.max_level())
        self.level = level

    def current_level(self) -> int:
        return self.level

    def max_level(self) -> int:
        return len(self.in_sizes) - 1

    @staticmethod
    def base_type() -> type[nn.Linear]:
        return nn.Linear

    def copy_to_base(self, dest: nn.Linear) -> None:
        dest.weight.data = self.linear.weight.data[:self.out_sizes[self.level],
                                                 :self.in_sizes[self.level]]
        if self.linear.bias is not None:
            dest.bias.data = self.linear.bias.data[:self.out_sizes[self.level]]

    def load_from_base(self, src: nn.Linear) -> None:
        self.linear.weight.data[:self.out_sizes[self.level],
                              :self.in_sizes[self.level]] = src.weight.data
        if src.bias is not None:
            self.linear.bias.data[:self.out_sizes[self.level]] = src.bias.data

    def make_base_copy(self) -> nn.Linear:
        lin = nn.Linear(
            self.in_sizes[self.level], self.out_sizes[self.level], *self._args, **self._kwargs)
        self.copy_to_base(lin)
        return lin
