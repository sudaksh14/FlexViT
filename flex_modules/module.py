from typing import Any, TypeVar, Generic

from torch import nn


T = TypeVar('T')


class LevelDelta(Generic[T]):
    def __init__(self, delta):
        self.delta: T = delta

    def verify_format():
        return True

    def to_device(self):
        return self

    def apply(self, module):
        raise NotImplemented()


class UpDelta(LevelDelta[T]):
    def __init__(self, delta):
        super().__init__(delta)

    def apply(self, module):
        return LevelDeltas.apply_level_delta_up(module, self)


class DownDelta(LevelDelta[T]):
    def __init__(self, delta):
        super().__init__(delta)

    def apply(self, module):
        return LevelDeltas.apply_level_delta_down(module, self)


class LevelDeltaCompatible:
    @staticmethod
    def base_type() -> type[nn.Module]:
        """
        Queries the type of regular module this flexible module is based on.
        """
        raise NotImplementedError()

    def export_level_delta(self) -> tuple[DownDelta, UpDelta]:
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
    def apply_level_delta_down(module: nn.Module, level_delta: DownDelta) -> None:
        """
        Takes regular layer and applies a delta down to it.
        """
        raise NotImplementedError()

    @staticmethod
    def apply_level_delta_up(module: nn.Module, level_delta: UpDelta) -> None:
        """
        Takes regular layer and applies a delta up to it.
        """
        raise NotImplementedError()

    @staticmethod
    def register_self(cls: type['LevelDeltaCompatible']) -> type['LevelDeltaCompatible']:
        return LevelDeltas.register(cls)


class Module(nn.Module, LevelDeltaCompatible):
    def __init__(self):
        nn.Module.__init__(self)
        LevelDeltaCompatible.__init__(self)

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


class LevelDeltas:
    registered: dict[type[nn.Module], LevelDeltaCompatible] = dict()

    @staticmethod
    def register(cls: type[LevelDeltaCompatible]) -> type[LevelDeltaCompatible]:
        assert (not __class__.is_registered(cls.base_type()))
        __class__.registered[cls.base_type()] = cls

    @staticmethod
    def is_registered(module_type: type[nn.Module]):
        return module_type in __class__.registered

    @staticmethod
    def apply_level_delta_down(module: nn.Module, level_delta: DownDelta) -> None:
        """
        Takes regular layer and applies a delta down to it.
        """
        if not __class__.is_registered(type(module)):
            raise KeyError()
        return __class__.registered[type(module)].apply_level_delta_down(module, level_delta)

    @staticmethod
    def apply_level_delta_up(module: nn.Module, level_delta: UpDelta) -> None:
        """
        Takes regular layer and applies a delta up to it.
        """
        if not __class__.is_registered(type(module)):
            raise KeyError()
        return __class__.registered[type(module)].apply_level_delta_up(module, level_delta)
