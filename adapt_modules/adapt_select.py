from torch import nn
import torch

from adapt_modules.module import Module


class AdaptSelect(Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        for level, l in enumerate(self.layers):
            self.add_module(f"level{level}", l)
        self.set_level_use(self.max_level())

    def current_layer(self) -> nn.Module:
        return self.layers[self.level]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.current_layer()(x)

    def set_level_use(self, level: int):
        self.level = level

    def current_level(self) -> int:
        return self.level

    def max_level(self) -> int:
        return len(self.layers) - 1

    def copy_to_base(self, dest: nn.Module):
        dest.load_state_dict(self.current_layer().state_dict())

    def load_from_base(self, src: nn.Module):
        self.current_layer().load_state_dict(src.state_dict())

    def export_level_delta(self):
        return self.layers[self.level].state_dict()

    @staticmethod
    def apply_level_delta_down(model: nn.Module, level_delta):
        model.load_state_dict(level_delta)

    @staticmethod
    def apply_level_delta_up(model: nn.Module, level_delta):
        model.load_state_dict(level_delta)
