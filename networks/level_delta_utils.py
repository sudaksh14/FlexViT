from torch import nn
from flex_modules.module import Module, LevelDelta, UpDelta, DownDelta

from networks.flex_model import FlexModel
from typing import Iterable, Any


def get_model_deltas(model: Module) -> dict[tuple[int, bool], LevelDelta]:
    deltas = dict()

    for level in range(model.max_level() + 1):
        model.set_level_use(level)
        delta_down, delta_up = model.export_level_delta()
        deltas[(level, False)] = delta_down
        deltas[(level, True)] = delta_up

    return deltas


class BaseDeltaManager:
    def get_level_delta(self, level: int, up: bool) -> LevelDelta:
        raise NotImplementedError()

    def move_model_to(self, model: nn.Module, current_level: int, target_level: int) -> None:
        if target_level > current_level:
            for i in range(current_level + 1, target_level + 1):
                self.get_level_delta(i, True).apply(model)
        elif target_level < current_level:
            for i in range(current_level - 1, target_level - 1, -1):
                self.get_level_delta(i, False).apply(model)


class InMemoryDeltaManager(BaseDeltaManager):
    def __init__(self, deltas: dict[tuple[int, bool], list[LevelDelta]]) -> None:
        super().__init__()
        self.deltas = deltas

    def get_level_delta(self, level: int, up: bool) -> Iterable[LevelDelta]:
        return self.deltas[(level, up)]
