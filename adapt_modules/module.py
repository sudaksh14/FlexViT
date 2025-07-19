from torch import nn
from typing import Union, Any
import torch
import copy


class Module(nn.Module):
    def set_level_use(self, level: int) -> None:
        raise NotImplementedError()

    def current_level(self) -> int:
        raise NotImplementedError()

    def max_level(self) -> int:
        raise NotImplementedError()

    @staticmethod
    def base_type() -> type[nn.Module]:
        raise NotImplementedError()

    def copy_to_base(self, dest: nn.Module) -> None:
        raise NotImplementedError()

    def load_from_base(self, src: nn.Module) -> None:
        raise NotImplementedError()

    def make_base_copy(self) -> nn.Module:
        raise NotImplementedError()

    def export_level_delta(self) -> tuple[Any, Any]:
        raise NotImplementedError()

    @staticmethod
    def apply_level_delta_down(model: nn.Module, level_delta: Any) -> None:
        raise NotImplementedError()

    @staticmethod
    def apply_level_delta_up(model: nn.Module, level_delta: Any) -> None:
        raise NotImplementedError()
