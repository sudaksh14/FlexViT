from torch import nn
import torch

from adapt_modules.module import Module
import torch.nn.functional as F

from typing import Iterable
import copy


class SelfAttention(Module):
    def __init__(self, hidden_dim: Iterable[int], num_heads: int, *args, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self._args = args
        self._kwargs = kwargs

        self.max_hidden_dim = self.hidden_dim[-1]

        self.level = self.max_level()
        self.attn = nn.MultiheadAttention(
            self.max_hidden_dim, self.num_heads, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, self.max_hidden_dim - self.hidden_dim[self.level]))
        x, _ = self.attn(x, x, x, need_weights=False)
        x = x[..., :self.hidden_dim[self.level]]
        return x

    def set_level_use(self, level: int) -> None:
        assert (level >= 0)
        assert (level <= self.max_level())
        self.level = level

    def current_level(self) -> int:
        return self.level

    def max_level(self) -> int:
        return len(self.hidden_dim) - 1

    @staticmethod
    def base_type() -> type[nn.MultiheadAttention]:
        return nn.MultiheadAttention

    def copy_to_base(self, dest: nn.MultiheadAttention) -> None:
        dest.in_proj_weight.data = self.attn.in_proj_weight.data[:]
        dest.in_proj_bias.data = self.attn.in_proj_bias.data[:]
        dest.out_proj.weight.data = self.attn.out_proj.weight.data[:]
        dest.out_proj.bias.data = self.attn.out_proj.bias.data[:]

    def load_from_base(self, src: nn.MultiheadAttention) -> None:
        self.attn.in_proj_weight.data[:] = src.in_proj_weight.data
        self.attn.in_proj_bias.data[:] = src.in_proj_bias.data
        self.attn.out_proj.weight.data[:] = src.out_proj.weight.data
        self.attn.out_proj.bias.data[:] = src.out_proj.bias.data

    def make_base_copy(self) -> nn.MultiheadAttention:
        lin = nn.MultiheadAttention(
            self.hidden_dim[self.level], self.num_heads, *self._args, **self._kwargs)
        self.copy_to_base(lin)
        return lin
