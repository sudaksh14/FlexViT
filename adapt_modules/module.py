from torch import nn
from typing import Union, Any


class Module(nn.Module):
    def set_level_use(self, level: int):
        raise NotImplemented()

    def current_level(self) -> int:
        raise NotImplemented()

    def max_level(self) -> int:
        raise NotImplemented()

    @staticmethod
    def base_type() -> type[nn.Module]:
        raise NotImplemented()

    def copy_to_base(self, dest: nn.Module):
        raise NotImplemented()

    def load_from_base(self, src: nn.Module):
        raise NotImplemented()

    def make_base_copy(self) -> nn.Module:
        raise NotImplemented()

    def export_level_delta(self) -> Union[Any, Any]:
        raise NotImplemented()

    @staticmethod
    def apply_level_delta_down(model: nn.Module, level_delta) -> None:
        raise NotImplemented()

    @staticmethod
    def apply_level_delta_up(model: nn.Module, level_delta) -> None:
        raise NotImplemented()
