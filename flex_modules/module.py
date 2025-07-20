from torch import nn
from typing import Union, Any
import torch
import copy


class Module(nn.Module):
    def set_level_use(self, level: int) -> None:
        """
        Sets the level the flexbible module will use.
        """
        raise NotImplementedError()

    def current_level(self) -> int:
        """
        Queries the level currently used by the Module.
        """
        raise NotImplementedError()

    def max_level(self) -> int:
        """
        Queries the highest level this module can be set to.
        """
        raise NotImplementedError()

    @staticmethod
    def base_type() -> type[nn.Module]:
        """
        Queries the type of regular module this flexible module is based on.
        """
        raise NotImplementedError()

    def copy_to_base(self, dest: nn.Module) -> None:
        """
        Copies the current level to a regular module.

        The type of regular module can be queried using Module.base_type()

        Note that the regular modules dimensions have to be compatibel with
        the dimensions of the current level.
        """
        raise NotImplementedError()

    def load_from_base(self, src: nn.Module) -> None:
        """
        Copies a regular module to the current level.

        The type of regular module can be queried using Module.base_type()

        Note that the regular modules dimensions have to be compatibel with
        the dimensions of the current level.

        Note that due to the nature of shared weights this action may also affect other levels.
        """
        raise NotImplementedError()

    def make_base_copy(self) -> nn.Module:
        """
        Makes a new regular layer with the current layer copied into it.
        """
        raise NotImplementedError()

    def export_level_delta(self) -> tuple[Any, Any]:
        """
        This function extracts part of the flexible layer into a down delta and an up delta. If
        you have a regular layer that is equivalent to one below the current
        level of the flexible layer. You can apply the up delta to it to make it equivalent
        to the current level. Similarly if you have a regular layer equivalent to one above the
        current level of the flexible layer, you can apply a delta down to make it equivalent
        to the current level.

        The functions to apply these deltas are apply_level_delta_down, and apply_level_delta_up
        """
        raise NotImplementedError()

    @staticmethod
    def apply_level_delta_down(module: nn.Module, level_delta: Any) -> None:
        """
        Takes regular layer and applies a delta down to it.
        """
        raise NotImplementedError()

    @staticmethod
    def apply_level_delta_up(module: nn.Module, level_delta: Any) -> None:
        """
        Takes regular layer and applies a delta up to it.
        """
        raise NotImplementedError()
