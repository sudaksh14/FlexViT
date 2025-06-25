from torch import nn
from typing import Union, Any, Iterable
import torch
import copy

import utils

from adapt_modules.module import Module


class ClassTokenLayer(Module):
    def __init__(self, hidden_dim: Iterable[int]):
        super().__init__()
        self.hidden_dims = hidden_dim
        self.token = nn.Parameter(torch.zeros(1, 1, hidden_dim[-1]))
        self.level = self.max_level()

    def forward(self, x, n):
        batch_class_token = self.token[
            :, :, :self.hidden_dims[self.level]].expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        return x

    def set_level_use(self, level: int) -> None:
        self.level = level

    def current_level(self) -> int:
        return self.level

    def max_level(self) -> int:
        return len(self.hidden_dims) - 1

    @staticmethod
    def base_type() -> type[nn.Module]:
        return utils.ClassTokenLayer

    def copy_to_base(self, dest: utils.ClassTokenLayer) -> None:
        dest.token.data = self.token.data[:, :, :self.hidden_dims[self.level]]

    def load_from_base(self, src: utils.ClassTokenLayer) -> None:
        self.token.data[:, :, :self.hidden_dims[self.level]] = src.token.data

    def make_base_copy(self) -> nn.Module:
        dest = utils.ClassTokenLayer(self.hidden_dims[self.level])
        self.copy_to_base(dest)
        return dest

    def export_level_delta(self) -> tuple[Any, Any]:
        raise NotImplemented()

    @staticmethod
    def apply_level_delta_down(model: nn.Module, level_delta: Any) -> None:
        raise NotImplemented()

    @staticmethod
    def apply_level_delta_up(model: nn.Module, level_delta: Any) -> None:
        raise NotImplemented()

    def get_frozen_params(self, level: int) -> Any:
        return None

    def restore_frozen_params(self, level: int, params: Any) -> None:
        return
