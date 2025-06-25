from torch import nn
import torch

from adapt_modules.module import Module
from adapt_modules.adapt_select import AdaptSelect


class BatchNorm2d(AdaptSelect):
    def __init__(self, channels, *args, **kwargs) -> None:
        self._args = args
        self._kwargs = kwargs
        self._channels = channels
        layers = [
            nn.BatchNorm2d(c, *args, **kwargs) for c in channels
        ]
        super().__init__(layers)

    @staticmethod
    def base_type() -> type[nn.BatchNorm2d]:
        return nn.BatchNorm2d

    def make_base_copy(self) -> nn.BatchNorm2d:
        m = nn.BatchNorm2d(
            self._channels[self.current_level()], *self._args, **self._kwargs)
        self.copy_to_base(m)
        return m

    def export_level_delta(self) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        return ((self.layers[self.level].weight.data, self.layers[self.level].bias.data), (self.layers[self.level].weight.data, self.layers[self.level].bias.data))

    @staticmethod
    def apply_level_delta_down(model: nn.BatchNorm2d, level_delta: tuple[torch.Tensor, torch.Tensor]):
        model.weight.data = level_delta[0][:]
        model.bias.data = level_delta[1][:]
        model.running_mean = None
        model.running_var = None

    @staticmethod
    def apply_level_delta_up(model: nn.BatchNorm2d, level_delta: tuple[torch.Tensor, torch.Tensor]):
        model.weight.data = level_delta[0][:]
        model.bias.data = level_delta[1][:]
        model.running_mean = None
        model.running_var = None
