from typing import Any

from torch import nn
import torch

import flex_modules as fm


class FlexModel(nn.Module):
    """
    Defines an interface followed by all flexible models.
    """

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
    def export_level_delta(self) -> Any:
        """
        Level deltas on the model level are used for some
        specific parts not covered by the deltas of the modules.
        """
        return None, None

    @staticmethod
    def apply_level_delta_down(model: nn.Module, level_delta: Any) -> None:
        """
        Takes regular layer and applies a delta down to it.
        """
        return

    @staticmethod
    def apply_level_delta_up(model: nn.Module, level_delta: Any) -> None:
        """
        Takes regular layer and applies a delta up to it.
        """
        return
