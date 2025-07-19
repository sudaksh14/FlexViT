from torch import nn
from typing import Union, Any, Iterable
import torch
import copy

import utils

from flex_modules.module import Module


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

    @torch.no_grad()
    def copy_to_base(self, dest: utils.PosEmbeddingLayer) -> None:
        dest.embedding.data = self.embedding.data[
            :, :, :self.hidden_dims[self.level]]

    @torch.no_grad()
    def load_from_base(self, src: utils.PosEmbeddingLayer) -> None:
        self.embedding.data[
            :, :, :self.hidden_dims[self.level]] = src.embedding.data

    @torch.no_grad()
    def make_base_copy(self) -> nn.Module:
        dest = utils.PosEmbeddingLayer(
            self.seq_length, self.hidden_dims[self.level])
        self.copy_to_base(dest)
        return dest

    @torch.no_grad()
    def export_level_delta(self) -> tuple[Any, Any]:
        return (
            self.hidden_dims[self.level],
            self.embedding[
                :, :,
                self.hidden_dims[self.level - 1]:self.hidden_dims[self.level]
            ]
        )

    @staticmethod
    @torch.no_grad()
    def apply_level_delta_down(model: utils.PosEmbeddingLayer, level_delta: Any) -> None:
        model.embedding.data = model.embedding.data[:, :, :level_delta]

    @staticmethod
    @torch.no_grad()
    def apply_level_delta_up(model: utils.PosEmbeddingLayer, level_delta: Any) -> None:
        model.embedding.data = torch.cat(
            [model.embedding.data, level_delta], dim=2)
