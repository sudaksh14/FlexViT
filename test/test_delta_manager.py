import unittest
import sys
import os

import torch
from torch import nn

import networks.level_delta_utils as levels
import networks.flexresnet as flexresnet
import networks.flexvgg as flexvgg
import networks.flexvit as flexvit
import networks.config
import utils


def randomize_params(net):
    for p in net.parameters():
        p.data = torch.rand(*p.shape) / 100
    return net


def make_randomized(func):
    def f(*args, **kwargs):
        net = func(*args, **kwargs)
        return randomize_params(net)
    return f


class TestDeltaManager():
    @staticmethod
    def check_equiv(a, b):
        if torch.isclose(a, b, rtol=1e-3, atol=1e-5).all():
            return True
        print(torch.isclose(a, b), file=sys.stderr)
        print(torch.abs(a - b))
        return False

    def get_flex_config(self) -> networks.config.FlexModelConfig:
        raise NotImplementedError()

    def make_input(self) -> torch.Tensor:
        raise NotImplementedError()

    def test_deltas(self):
        aconfig = self.get_flex_config()
        model = aconfig.make_model()
        randomize_params(model)

        model.set_level_use(model.max_level())
        delta_manager = levels.InMemoryDeltaManager(model, model.max_level())
        reg_model = delta_manager.managed_model()

        model.eval()
        reg_model.eval()

        x = self.make_input()
        self.assertTrue(self.check_equiv(model(x), reg_model(x)))

        for i in range(model.max_level() - 1, -1, -1):
            model.set_level_use(i)
            delta_manager.move_to(i)
            self.assertTrue(self.check_equiv(model(x), reg_model(x)))

        for i in range(1, model.max_level() + 1):
            model.set_level_use(i)
            delta_manager.move_to(i)
            self.assertTrue(self.check_equiv(model(x), reg_model(x)))

    def test_deltas_file_deltas(self):
        filename = "temp.pt"
        try:
            config = self.get_flex_config()
            model = config.make_model()
            model.eval()

            with open(filename, "wb") as f:
                levels.FileDeltaManager.make_delta_file(f, model)

            x = self.make_input()

            reg_config = config.create_base_config(model.current_level())
            with levels.file_delta_manager(filename, reg_config) as manager:
                reg_model = manager.managed_model()
                reg_model.eval()

                self.assertTrue(self.check_equiv(model(x), reg_model(x)))

                for i in range(model.max_level() - 1, -1, -1):
                    model.set_level_use(i)
                    manager.move_to(i)
                    self.assertTrue(self.check_equiv(model(x), reg_model(x)))

                for i in range(1, model.max_level() + 1):
                    model.set_level_use(i)
                    manager.move_to(i)
                    self.assertTrue(self.check_equiv(model(x), reg_model(x)))
        finally:
            os.remove(filename)


class TestDeltaResnet(TestDeltaManager, unittest.TestCase):
    def get_flex_config(self) -> networks.config.FlexModelConfig:
        return flexresnet.ResnetConfig()

    def make_input(self) -> torch.Tensor:
        return torch.rand(1, 3, 32, 32)


class TestDeltaViT(TestDeltaManager, unittest.TestCase):
    def get_flex_config(self) -> networks.config.FlexModelConfig:
        return flexvit.ViTConfig()

    def make_input(self) -> torch.Tensor:
        return torch.rand(1, 3, 224, 224)


class TestDeltaVGG(TestDeltaManager, unittest.TestCase):
    def get_flex_config(self) -> networks.config.FlexModelConfig:
        return flexvgg.VGGConfig().no_prebuilt()

    def make_input(self) -> torch.Tensor:
        return torch.rand(1, 3, 32, 32)
