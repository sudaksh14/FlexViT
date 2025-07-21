import unittest

import torch

import networks.level_delta_utils as levels
import networks.flexresnet as flexresnet
import networks.flexvgg as flexvgg
import networks.flexvit as flexvit
import networks.config
import utils


class TestDeltaManager():
    @staticmethod
    def check_equiv(a, b, error=1e5):
        return (torch.abs(a - b) < error).all()

    def get_flex_config(self) -> networks.config.FlexModelConfig:
        raise NotImplementedError()

    def make_input(self) -> torch.Tensor:
        raise NotImplementedError()

    def test_deltas(self):
        aconfig = self.get_flex_config()
        model = aconfig.make_model().to(utils.get_device())
        model.set_level_use(model.max_level())
        reg_model = aconfig.create_base_config(
            model.current_level()).make_model().to(utils.get_device())
        utils.flexible_model_copy(model, reg_model)

        x = self.make_input()
        self.assertTrue(self.check_equiv(model(x), reg_model(x)))

        deltas = levels.get_model_deltas(model)
        delta_manager = levels.InMemoryDeltaManager(deltas)

        for i in range(model.max_level() - 1, -1, -1):
            model.set_level_use(i)
            delta_manager.move_model_to(reg_model, i + 1, i)
            self.assertTrue(self.check_equiv(model(x), reg_model(x)))

        for i in range(1, model.max_level() + 1):
            model.set_level_use(i)
            delta_manager.move_model_to(reg_model, i - 1, i)
            self.assertTrue(self.check_equiv(model(x), reg_model(x)))


class TestDeltaResnet(TestDeltaManager, unittest.TestCase):
    def get_flex_config(self) -> networks.config.FlexModelConfig:
        return flexresnet.ResnetConfig()

    def make_input(self) -> torch.Tensor:
        return torch.rand(10, 3, 32, 32, device=utils.get_device())


class TestDeltaViT(TestDeltaManager, unittest.TestCase):
    def get_flex_config(self) -> networks.config.FlexModelConfig:
        return flexvit.ViTConfig()

    def make_input(self) -> torch.Tensor:
        return torch.rand(
            10, 3, 224, 224, device=utils.get_device())


class TestDeltaVGG(TestDeltaManager, unittest.TestCase):
    def get_flex_config(self) -> networks.config.FlexModelConfig:
        return flexvgg.VGGConfig()

    def make_input(self) -> torch.Tensor:
        return torch.rand(10, 3, 32, 32, device=utils.get_device())
