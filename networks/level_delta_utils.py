from torch import nn
from flex_modules.module import Module, LevelDeltas

from networks.flex_model import FlexModel
from typing import Iterable, Any


def get_model_deltas(model: FlexModel) -> dict[tuple[int, bool], list[Any]]:
    deltas = dict()

    for level in range(model.max_level() + 1):
        delta_downs = []
        delta_ups = []
        model.set_level_use(level)
        for module in model.modules():
            if not isinstance(module, Module):
                if not isinstance(module, FlexModel):
                    continue
            delta_down, delta_up = module.export_level_delta()
            delta_downs.append(delta_down)
            delta_ups.append(delta_up)

        deltas[(level, False)] = delta_downs
        deltas[(level, True)] = delta_ups

    return deltas


def apply_delta_down(model: nn.Module, deltas: Iterable[Any]) -> None:
    dest_it = iter(model.modules())
    for delta in deltas:
        while True:
            module = next(dest_it)
            if not LevelDeltas.is_registered(type(module)):
                continue
            LevelDeltas.apply_level_delta_down(module, delta)
            break


def apply_delta_up(model: nn.Module, deltas: Iterable[Any]) -> None:
    dest_it = iter(model.modules())
    for delta in deltas:
        while True:
            module = next(dest_it)
            if not LevelDeltas.is_registered(type(module)):
                continue
            LevelDeltas.apply_level_delta_up(module, delta)
            break


class BaseDeltaManager:
    def get_level_delta(self, level: int, up: bool) -> Iterable[Any]:
        raise NotImplementedError()

    def move_model_to(self, model: nn.Module, current_level: int, target_level: int) -> None:
        if target_level > current_level:
            for i in range(current_level + 1, target_level + 1):
                apply_delta_up(model, self.get_level_delta(
                    i, True))
        elif target_level < current_level:
            for i in range(current_level - 1, target_level - 1, -1):
                apply_delta_down(model, self.get_level_delta(
                    i, False))


class InMemoryDeltaManager(BaseDeltaManager):
    def __init__(self, deltas: dict[tuple[int, bool], list[Any]]) -> None:
        super().__init__()
        self.deltas = deltas

    def get_level_delta(self, level: int, up: bool) -> Iterable[Any]:
        return self.deltas[(level, up)]
