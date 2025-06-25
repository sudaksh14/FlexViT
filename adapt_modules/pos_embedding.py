from torch import nn
from typing import Union, Any, Iterable
import torch
import copy

import utils

from adapt_modules.module import Module


class PosEmbeddingLayer(Module):
    def __init__(self, seq_length, hidden_dim: Iterable[int]):
        super().__init__()
        self.seq_length = seq_length
        self.hidden_dims = hidden_dim
        self.embedding = nn.Parameter(torch.empty(
            1, seq_length, hidden_dim[-1]).normal_(std=0.02))
        self.level = self.max_level()

    def forward(self, x):
        return x + self.embedding[
            :, :, :self.hidden_dims[self.level]]

    def set_level_use(self, level: int) -> None:
        self.level = level

    def current_level(self) -> int:
        return self.level

    def max_level(self) -> int:
        return len(self.hidden_dims) - 1

    @staticmethod
    def base_type() -> type[nn.Module]:
        return utils.PosEmbeddingLayer

    def copy_to_base(self, dest: utils.PosEmbeddingLayer) -> None:
        dest.embedding.data = self.embedding.data[
            :, :, :self.hidden_dims[self.level]]

    def load_from_base(self, src: utils.PosEmbeddingLayer) -> None:
        self.embedding.data[
            :, :, :self.hidden_dims[self.level]] = src.embedding.data

    def make_base_copy(self) -> nn.Module:
        dest = utils.PosEmbeddingLayer(
            self.seq_length, self.hidden_dims[self.level])
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
