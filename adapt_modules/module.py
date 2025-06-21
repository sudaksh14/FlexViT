from torch import nn
from typing import Union, Any


class Module(nn.Module):
    def set_level_use(self, level: int) -> None:
        raise NotImplemented()

    def current_level(self) -> int:
        raise NotImplemented()

    def max_level(self) -> int:
        raise NotImplemented()

    @staticmethod
    def base_type() -> type[nn.Module]:
        raise NotImplemented()

    def copy_to_base(self, dest: nn.Module) -> None:
        raise NotImplemented()

    def load_from_base(self, src: nn.Module) -> None:
        raise NotImplemented()

    def make_base_copy(self) -> nn.Module:
        raise NotImplemented()

    def export_level_delta(self) -> tuple[Any, Any]:
        raise NotImplemented()

    @staticmethod
    def apply_level_delta_down(model: nn.Module, level_delta: Any) -> None:
        raise NotImplemented()

    @staticmethod
    def apply_level_delta_up(model: nn.Module, level_delta: Any) -> None:
        raise NotImplemented()

    def zero_out_gradients(self, level: int) -> None:
        pass
