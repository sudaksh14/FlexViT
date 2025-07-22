import unittest

from torch import nn
import torch

from flex_modules.module import Module
import flex_modules as fm
import utils


class LayerTester:
    def make_flex_module(self) -> Module:
        raise NotImplementedError()

    def make_reg_module(self, level) -> nn.Module:
        raise NotImplementedError()

    def make_input(self, level) -> torch.Tensor:
        raise NotImplementedError()

    def augment_for_reg(self, x, level) -> torch.Tensor:
        return x

    def augment_reg_output(self, y, level) -> torch.Tensor:
        return y

    @staticmethod
    def check_equiv(a, b, error=1e5) -> bool:
        return (torch.abs(a - b) < error).all()

    def test_basic_forward(self):
        layer = self.make_flex_module()
        for i in range(layer.max_level() + 1):
            x = self.make_input(i)
            layer.set_level_use(i)
            layer(*x)

    def test_copy_to_base(self):
        layer = self.make_flex_module()
        for i in range(layer.max_level() + 1):
            layer.set_level_use(i)
            reg_layer = self.make_reg_module(i)
            layer.copy_to_base(reg_layer)
            x = self.make_input(i)
            self.assertTrue(self.check_equiv(
                layer(*x), self.augment_reg_output(reg_layer(*self.augment_for_reg(x, i)), i))
            )

    def test_load_from_base(self):
        layer = self.make_flex_module()
        for i in range(layer.max_level() + 1):
            layer.set_level_use(i)
            reg_layer = self.make_reg_module(i)
            layer.load_from_base(reg_layer)
            x = self.make_input(i)
            self.assertTrue(self.check_equiv(
                layer(*x), self.augment_reg_output(reg_layer(*self.augment_for_reg(x, i)), i)))

    def test_make_base_copy(self):
        layer = self.make_flex_module()
        for i in range(layer.max_level() + 1):
            layer.set_level_use(i)
            reg_layer = layer.make_base_copy()
            x = self.make_input(i)
            self.assertTrue(self.check_equiv(
                layer(*x), self.augment_reg_output(reg_layer(*self.augment_for_reg(x, i)), i)))

    def test_level_deltas(self):
        layer = self.make_flex_module()
        layer.set_level_use(layer.max_level())
        reg_layer = self.make_reg_module(layer.max_level())
        layer.copy_to_base(reg_layer)

        x = self.make_input(layer.max_level())
        self.assertTrue(self.check_equiv(
            layer(*x), self.augment_reg_output(
                reg_layer(
                    *self.augment_for_reg(
                        x, layer.max_level()
                    )
                ),
                layer.max_level())))

        for i in range(layer.max_level() - 1, -1, -1):
            layer.set_level_use(i)
            delta_down, delta_up = layer.export_level_delta()
            layer.apply_level_delta_down(reg_layer, delta_down)
            x = self.make_input(i)
            self.assertTrue(self.check_equiv(
                layer(*x), self.augment_reg_output(reg_layer(*self.augment_for_reg(x, i)), i)))

        for i in range(1, layer.max_level() + 1):
            layer.set_level_use(i)
            delta_down, delta_up = layer.export_level_delta()
            layer.apply_level_delta_up(reg_layer, delta_up)
            x = self.make_input(i)
            self.assertTrue(self.check_equiv(
                layer(*x), self.augment_reg_output(reg_layer(*self.augment_for_reg(x, i)), i)))

    def test_level_deltas_LevelDeltas(self):
        layer = self.make_flex_module()
        layer.set_level_use(layer.max_level())
        reg_layer = self.make_reg_module(layer.max_level())
        layer.copy_to_base(reg_layer)

        x = self.make_input(layer.max_level())
        self.assertTrue(self.check_equiv(
            layer(*x), self.augment_reg_output(
                reg_layer(
                    *self.augment_for_reg(
                        x, layer.max_level()
                    )
                ),
                layer.max_level())))

        for i in range(layer.max_level() - 1, -1, -1):
            layer.set_level_use(i)
            delta_down, delta_up = layer.export_level_delta()
            delta_down.apply(reg_layer)
            x = self.make_input(i)
            self.assertTrue(self.check_equiv(
                layer(*x), self.augment_reg_output(reg_layer(*self.augment_for_reg(x, i)), i)))

        for i in range(1, layer.max_level() + 1):
            layer.set_level_use(i)
            delta_down, delta_up = layer.export_level_delta()
            delta_up.apply(reg_layer)
            x = self.make_input(i)
            self.assertTrue(self.check_equiv(
                layer(*x), self.augment_reg_output(reg_layer(*self.augment_for_reg(x, i)), i)))


class TestConv2d(LayerTester, unittest.TestCase):
    IN_CHANNELS = [5, 9, 12]
    OUT_CHANNELS = [5, 9, 12]

    def make_flex_module(self) -> Module:
        return fm.conv2d.Conv2d(
            self.IN_CHANNELS, self.OUT_CHANNELS, kernel_size=3, bias=True)

    def make_reg_module(self, level) -> nn.Module:
        return nn.Conv2d(
            self.IN_CHANNELS[level], self.OUT_CHANNELS[level], kernel_size=3, bias=True)

    def make_input(self, level) -> torch.Tensor:
        return torch.rand(10, self.IN_CHANNELS[level], 100, 100),


class TestAttention(LayerTester, unittest.TestCase):
    TOKEN_SIZE = [25, 30, 56, 100]
    HEADS = [5, 6, 8, 10]

    def make_flex_module(self) -> Module:
        return fm.SelfAttention(self.TOKEN_SIZE, self.HEADS)

    def make_reg_module(self, level) -> nn.Module:
        return nn.MultiheadAttention(
            self.TOKEN_SIZE[level],
            self.HEADS[level],
            batch_first=True)

    def make_input(self, level) -> torch.Tensor:
        return torch.rand((10, 20, self.TOKEN_SIZE[level])),

    def augment_for_reg(self, x, level) -> torch.Tensor:
        res = x[0], x[0], x[0]
        return res

    def augment_reg_output(self, y, level) -> torch.Tensor:
        a, b = y
        return a


class TestLinear(LayerTester, unittest.TestCase):
    IN_SIZE = [10, 20, 25, 40]
    OUT_SIZE = [5, 15, 30, 50]

    def make_flex_module(self) -> Module:
        return fm.Linear(self.IN_SIZE, self.OUT_SIZE)

    def make_reg_module(self, level) -> nn.Module:
        return nn.Linear(self.IN_SIZE[level], self.OUT_SIZE[level])

    def make_input(self, level) -> torch.Tensor:
        return torch.rand((100, self.IN_SIZE[level])),


class TestLayerNorm(LayerTester, unittest.TestCase):
    TOKEN_SIZE = [25, 30, 56, 100]

    def make_flex_module(self):
        return fm.LayerNorm(self.TOKEN_SIZE, eps=1e-6)

    def make_reg_module(self, level):
        return nn.LayerNorm(self.TOKEN_SIZE[level], eps=1e-6)

    def make_input(self, level):
        return torch.rand(10, 100, self.TOKEN_SIZE[level]),


class TestPosEmbedding(LayerTester, unittest.TestCase):
    TOKEN_SIZE = [25, 30, 56, 100]

    def make_flex_module(self):
        return fm.PosEmbeddingLayer(100, self.TOKEN_SIZE)

    def make_reg_module(self, level):
        return utils.PosEmbeddingLayer(100, self.TOKEN_SIZE[level])

    def make_input(self, level):
        return torch.rand(10, 100, self.TOKEN_SIZE[level]),


class TestLinearSelect(LayerTester, unittest.TestCase):
    IN_SIZE = [10, 20, 25, 40]
    OUT_SIZE = [5, 15, 30, 50]

    def make_flex_module(self) -> Module:
        return fm.LinearSelect(self.IN_SIZE, self.OUT_SIZE)

    def make_reg_module(self, level) -> nn.Module:
        return utils.LinearHead(self.IN_SIZE[level], self.OUT_SIZE[level])

    def make_input(self, level) -> torch.Tensor:
        return torch.rand((100, self.IN_SIZE[level])),


class TestClassToken(LayerTester, unittest.TestCase):
    TOKEN_SIZE = [25, 30, 56, 100]

    def make_flex_module(self):
        return fm.ClassTokenLayer(self.TOKEN_SIZE)

    def make_reg_module(self, level):
        return utils.ClassTokenLayer(self.TOKEN_SIZE[level])

    def make_input(self, level):
        return torch.rand(10, 100, self.TOKEN_SIZE[level]), 10


class TestBatchnorm2d(LayerTester, unittest.TestCase):
    CHANNELS = [5, 9, 12]

    def make_flex_module(self) -> Module:
        return fm.BatchNorm2d(self.CHANNELS)

    def make_reg_module(self, level) -> nn.Module:
        return nn.BatchNorm2d(self.CHANNELS[level])

    def make_input(self, level) -> torch.Tensor:
        return torch.rand(10, self.CHANNELS[level], 100, 100),
