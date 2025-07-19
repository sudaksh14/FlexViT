from torch import nn
import adapt_modules as am

from typing import Any
import torch


class AdaptModel(nn.Module):
    def set_level_use(self, level) -> None:
        self.level = level
        for _, module in self.named_modules():
            if isinstance(module, am.Module):
                module.set_level_use(level)

    def current_level(self) -> int:
        raise NotImplementedError()

    def max_level(self) -> int:
        raise NotImplementedError()

    @staticmethod
    def base_type() -> type[nn.Module]:
        raise NotImplementedError()

    @torch.no_grad()
    def export_level_delta(self) -> Any:
        return None, None

    @staticmethod
    def apply_level_delta_down(model: nn.Module, level_delta: Any) -> None:
        return

    @staticmethod
    def apply_level_delta_up(model: nn.Module, level_delta: Any) -> None:
        return
