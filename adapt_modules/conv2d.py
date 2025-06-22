from torch import nn
import torch

from adapt_modules.module import Module
import torch.nn.functional as F

from typing import Iterable
import copy


class Conv2d(Module):
    def __init__(self, in_sizes: Iterable[int], out_sizes: Iterable[int], *args, **kwargs) -> None:
        super().__init__()
        self.in_sizes = in_sizes
        self.out_sizes = out_sizes

        self._args = args
        self._kwargs = kwargs

        assert (not kwargs.get('bias', False))
        assert (len(self.in_sizes) > 0)
        assert (len(self.out_sizes) > 0)
        assert (len(self.in_sizes) == len(self.out_sizes))
        assert (sorted(self.in_sizes) == list(self.in_sizes))
        assert (sorted(self.out_sizes) == list(self.out_sizes))
        assert (self.in_sizes[0] > 0)
        assert (self.out_sizes[0] > 0)

        self.max_in_size = self.in_sizes[-1]
        self.max_out_size = self.out_sizes[-1]

        self._cached = (-1, -1, -1)
        self._zeros_cache = None

        self.set_level_use(self.max_level())
        kwargs['bias'] = False
        self.conv = nn.Conv2d(
            self.max_in_size, self.max_out_size, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 0, 0, 0, 0,
                      self.max_in_size - self.in_sizes[self.level]))
        x = self.conv(x)
        x = x[:, :self.out_sizes[self.level]]
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
    def base_type() -> type[nn.Conv2d]:
        return nn.Conv2d

    def copy_to_base(self, dest: nn.Conv2d) -> None:
        dest.weight.data = self.conv.weight.data[:self.out_sizes[self.level],
                                                 :self.in_sizes[self.level]]

    def load_from_base(self, src: nn.Conv2d) -> None:
        self.conv.weight.data[:self.out_sizes[self.level],
                              :self.in_sizes[self.level]] = src.weight.data

    def make_base_copy(self) -> nn.Conv2d:
        conv = nn.Conv2d(
            self.in_sizes[self.level], self.out_sizes[self.level], *self._args, **self._kwargs)
        self.copy_to_base(conv)
        return conv

    def export_level_delta(self) -> tuple[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]:
        weights = self.conv.weight.data
        lower_part = weights[:self.out_sizes[self.level],
                             self.in_sizes[self.level-1]:self.in_sizes[self.level], ]
        right_part = weights[self.out_sizes[self.level-1]:self.out_sizes[self.level], :self.in_sizes[self.level-1]]
        prune_up = (lower_part, right_part)
        prune_down = (self.in_sizes[self.level], self.out_sizes[self.level])
        return prune_down, prune_up

    @staticmethod
    def apply_level_delta_down(model: nn.Conv2d, level_delta: tuple[int, int]) -> None:
        in_size, out_size = level_delta
        model.weight.data = model.weight.data[:out_size, :in_size]

    @staticmethod
    def apply_level_delta_up(model: nn.Conv2d, level_delta: tuple[torch.Tensor, torch.Tensor]) -> None:
        weights = model.weight.data
        lower_part, right_part = level_delta
        out_size, in_size, *_ = weights.size()
        weights = F.pad(
            weights, (0, 0, 0, 0, 0, lower_part.size(1), 0, right_part.size(0)))
        weights[:, in_size:] = lower_part
        weights[out_size:, :in_size] = right_part
        model.weight.data = weights

        model.zero_grad()

    def get_frozen_params(self, level: int):
        if level < 0:
            return None
        cpy = self.conv.weight.data[:self.out_sizes[level],
                                    :self.in_sizes[level]]
        cpy = copy.deepcopy(cpy)
        cpy = cpy.detach()
        return cpy

    def restore_frozen_params(self, level: int, params) -> None:
        if level < 0:
            return
        self.conv.weight.data[:self.out_sizes[level],
                              :self.in_sizes[level]] = params
