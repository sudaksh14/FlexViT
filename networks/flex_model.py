from typing import Any

from torch import nn
import torch

import flex_modules as fm


class FlexModel(nn.Module, fm.LevelDeltaCompatible):
    """
    Defines an interface followed by all flexible models.
    """

    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        fm.LevelDeltaCompatible.__init__(self)

    def set_level_use(self, level) -> None:
        """
        Sets the level the flexbible model will use.
        """
        self.level = level
        for _, module in self.named_modules():
            if isinstance(module, fm.Module):
                module.set_level_use(level)

    def current_level(self) -> int:
        """
        Queries the level currently used by the model.
        """
        raise NotImplementedError()

    def max_level(self) -> int:
        """
        Queries the highest level this model can be set to.
        """
        raise NotImplementedError()

    @staticmethod
    def base_type() -> type[nn.Module]:
        """
        Queries the type of regular module this flexible module is based on.
        """
        raise NotImplementedError()

    @torch.no_grad()
    def export_level_delta(self) -> tuple[fm.DownDelta, fm.UpDelta]:
        """
        Level deltas on the model level are used for some
        specific parts not covered by the deltas of the modules.
        """
        delta_downs = []
        delta_ups = []
        for module in self.modules():
            if module is self:
                continue
            if not isinstance(module, fm.LevelDeltaCompatible):
                continue
            delta_down, delta_up = module.export_level_delta()
            delta_downs.append(delta_down)
            delta_ups.append(delta_up)
        delta_downs = tuple(delta_downs)
        delta_ups = tuple(delta_ups)
        return fm.DownDelta(delta_downs), fm.UpDelta(delta_ups)

    @staticmethod
    def apply_level_delta_down(model: nn.Module, level_delta: fm.DownDelta[tuple[fm.DownDelta, ...]]) -> None:
        """
        Takes regular layer and applies a delta down to it.
        """
        dest_it = iter(model.modules())
        for delta in level_delta.delta:
            while True:
                module = next(dest_it)
                if module is model:
                    continue

                try:
                    delta.apply(module)
                    break
                except KeyError:
                    pass

    @staticmethod
    def apply_level_delta_up(model: nn.Module, level_delta: fm.DownDelta[tuple[fm.UpDelta, ...]]) -> None:
        """
        Takes regular layer and applies a delta up to it.
        """
        dest_it = iter(model.modules())
        for delta in level_delta.delta:
            while True:
                module = next(dest_it)
                if module is model:
                    continue
                try:
                    delta.apply(module)
                    break
                except KeyError:
                    pass
