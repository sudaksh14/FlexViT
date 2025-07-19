from torch import nn
import torch

from flex_modules.module import Module
from flex_modules.flexselect import AdaptSelect


class LayerNorm(AdaptSelect):
    def __init__(self, channels, *args, **kwargs) -> None:
        self._args = args
        self._kwargs = kwargs
        self._channels = channels
        layers = [
            nn.LayerNorm(c, *args, **kwargs) for c in channels
        ]
        super().__init__(layers)

    @staticmethod
    def base_type() -> type[nn.LayerNorm]:
        return nn.LayerNorm

    @torch.no_grad()
    def make_base_copy(self) -> nn.LayerNorm:
        m = nn.LayerNorm(
            self._channels[self.current_level()], *self._args, **self._kwargs)
        self.copy_to_base(m)
        return m

    @staticmethod
    @torch.no_grad()
    def apply_level_delta_down(model: nn.LayerNorm, level_delta: tuple[torch.Tensor, torch.Tensor]):
        AdaptSelect.apply_level_delta_down(model, level_delta)
        model.normalized_shape = level_delta[0].shape[0],

    @staticmethod
    @torch.no_grad()
    def apply_level_delta_up(model: nn.LayerNorm, level_delta: tuple[torch.Tensor, torch.Tensor]):
        AdaptSelect.apply_level_delta_up(model, level_delta)
        model.normalized_shape = level_delta[0].shape[0],
