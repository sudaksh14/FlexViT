from typing import Iterable, Any

from torch import nn
import torch.nn.functional as F
import torch

from flex_modules.module import Module, DownDelta, UpDelta


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

    def export_level_delta(self) -> tuple[DownDelta[tuple[int, int]], UpDelta[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        weights = self.linear.weight.data
        lower_part = weights[:self.out_sizes[self.level],
                             self.in_sizes[self.level-1]:self.in_sizes[self.level], ]
        right_part = weights[self.out_sizes[self.level-1]:self.out_sizes[self.level], :self.in_sizes[self.level-1]]
        bias_part = None
        if self.linear.bias is not None:
            bias_part = self.linear.bias.data[
                self.out_sizes[self.level-1]:self.out_sizes[self.level]]
        prune_up = (lower_part, right_part, bias_part)
        prune_down = (self.in_sizes[self.level], self.out_sizes[self.level])
        return DownDelta(prune_down), UpDelta(prune_up)

    @staticmethod
    def apply_level_delta_down(model: nn.Module, level_delta: DownDelta[tuple[int, int]]) -> None:
        in_size, out_size = level_delta.delta
        model.weight.data = model.weight.data[:out_size, :in_size]
        if model.bias is not None:
            model.bias.data = model.bias.data[:out_size]

    @staticmethod
    def apply_level_delta_up(model: nn.Module, level_delta: UpDelta[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> None:
        weights = model.weight.data
        lower_part, right_part, bias_part = level_delta.delta
        out_size, in_size, *_ = weights.size()
        weights = F.pad(
            weights, (0, lower_part.size(1), 0, right_part.size(0)))
        weights[:, in_size:] = lower_part
        weights[out_size:, :in_size] = right_part
        if model.bias is not None:
            model.bias.data = torch.cat([model.bias.data, bias_part])
        model.weight.data = weights
        model.zero_grad()


Linear.register_self(Linear)
