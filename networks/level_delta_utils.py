from torch import nn
from adapt_modules.module import Module
import torch

from networks.adapt_model import AdaptModel
from typing import Iterable, Any


def get_model_deltas(model: AdaptModel) -> tuple[dict[tuple[int, bool], list[Any]], list[type[Module]]]:
    module_type_list = []
    deltas = dict()
    for module in model.modules():
        if not isinstance(module, Module):
            if not isinstance(module, AdaptModel):
                continue
        module_type_list.append(type(module))
    for level in range(model.max_level() + 1):
        delta_downs = []
        delta_ups = []
        model.set_level_use(level)
        for module in model.modules():
            if not isinstance(module, Module):
                if not isinstance(module, AdaptModel):
                    continue
            delta_down, delta_up = module.export_level_delta()
            delta_downs.append(delta_down)
            delta_ups.append(delta_up)

        deltas[(level, False)] = delta_downs
        deltas[(level, True)] = delta_ups

    return deltas, module_type_list


def apply_delta_down(model: nn.Module, deltas: Iterable[Any], module_type_list: Iterable[type[Module]]) -> None:
    dest_it = iter(model.modules())
    for delta, module_type in zip(deltas, module_type_list):
        while True:
            module = next(dest_it)
            if not isinstance(module, module_type.base_type()):
                continue
            module_type.apply_level_delta_down(module, delta)
            break


def apply_delta_up(model: nn.Module, deltas: Iterable[Any], module_type_list: Iterable[type[Module]]) -> None:
    dest_it = iter(model.modules())
    for delta, module_type in zip(deltas, module_type_list):
        while True:
            module = next(dest_it)
            if not isinstance(module, module_type.base_type()):
                continue
            module_type.apply_level_delta_up(module, delta)
            break


class BaseDeltaManager:
    def get_module_list(self) -> Iterable[type[Module]]:
        raise NotImplementedError()

    def get_level_delta(self, level: int, up: bool) -> Iterable[Any]:
        raise NotImplementedError()

    def move_model_to(self, model: nn.Module, current_level: int, target_level: int) -> None:
        if target_level > current_level:
            for i in range(current_level + 1, target_level + 1):
                apply_delta_up(model, self.get_level_delta(
                    i, True), self.get_module_list())
        elif target_level < current_level:
            for i in range(current_level - 1, target_level - 1, -1):
                apply_delta_down(model, self.get_level_delta(
                    i, False), self.get_module_list())


class InMemoryDeltaManager(BaseDeltaManager):
    def __init__(self, deltas: dict[tuple[int, bool], list[Any]], module_type_list: list[type[Module]]) -> None:
        super().__init__()
        self.deltas = deltas
        self.module_type_lists = module_type_list

    def get_module_list(self) -> Iterable[type[Module]]:
        return self.module_type_lists

    def get_level_delta(self, level: int, up: bool) -> Iterable[Any]:
        return self.deltas[(level, up)]
