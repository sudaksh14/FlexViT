import unittest
import adapt_modules as am

from torch import nn
import torch

from adapt_modules.module import Module
import torch.nn.functional as F


class TestConv2d(unittest.TestCase):
    IN_CHANNELS = [5, 9, 12]
    OUT_CHANNELS = [5, 9, 12]

    def setUp(self):
        self.aconv = am.Conv2d(
            self.IN_CHANNELS, self.OUT_CHANNELS, kernel_size=3, bias=True)
        self.rand_init_conv()

    def rand_init_conv(self):
        for p in self.aconv.parameters():
            p.data[:] = torch.rand(*p.data.size())

    @staticmethod
    def check_equiv(a, b, error=1e5):
        return (torch.abs(a - b) < error).all()

    def test_basic_forward(self):
        for i in range(self.aconv.max_level()+1):
            x = torch.zeros(10, self.IN_CHANNELS[i], 100, 100)
            self.aconv.set_level_use(i)
            self.aconv(x)

    def test_copy_to_base(self):
        for i in range(self.aconv.max_level()+1):
            self.rand_init_conv()
            self.aconv.set_level_use(i)
            conv = nn.Conv2d(
                self.IN_CHANNELS[i], self.OUT_CHANNELS[i], kernel_size=3, bias=True)
            self.aconv.copy_to_base(conv)
            x = torch.rand(10, self.IN_CHANNELS[i], 100, 100)
            self.assertTrue(self.check_equiv(self.aconv(x), conv(x)))

    def test_load_from_base(self):
        for i in range(self.aconv.max_level()+1):
            self.rand_init_conv()
            self.aconv.set_level_use(i)
            conv = nn.Conv2d(
                self.IN_CHANNELS[i], self.OUT_CHANNELS[i], kernel_size=3, bias=True)
            self.aconv.load_from_base(conv)
            x = torch.rand(10, self.IN_CHANNELS[i], 100, 100)
            self.assertTrue(self.check_equiv(self.aconv(x), conv(x)))

    def test_make_base_copy(self):
        for i in range(self.aconv.max_level()+1):
            self.rand_init_conv()
            self.aconv.set_level_use(i)
            conv = self.aconv.make_base_copy()
            x = torch.rand(10, self.IN_CHANNELS[i], 100, 100)
            self.assertTrue(self.check_equiv(self.aconv(x), conv(x)))

    def test_level_deltas(self):
        self.aconv.set_level_use(self.aconv.max_level())
        conv = nn.Conv2d(
            self.IN_CHANNELS[-1], self.OUT_CHANNELS[-1], kernel_size=3, bias=True)
        self.aconv.copy_to_base(conv)

        x = torch.rand(10, self.IN_CHANNELS[-1], 100, 100)
        self.assertTrue(self.check_equiv(self.aconv(x), conv(x)))

        for i in range(len(self.IN_CHANNELS) - 2, -1, -1):
            self.aconv.set_level_use(i)
            delta_down, delta_up = self.aconv.export_level_delta()
            self.aconv.apply_level_delta_down(conv, delta_down)
            x = torch.rand(10, self.IN_CHANNELS[i], 100, 100)
            self.assertTrue(self.check_equiv(self.aconv(x), conv(x)))

        for i in range(1, len(self.IN_CHANNELS)):
            self.aconv.set_level_use(i)
            delta_down, delta_up = self.aconv.export_level_delta()
            self.aconv.apply_level_delta_up(conv, delta_up)
            x = torch.rand(10, self.IN_CHANNELS[i], 100, 100)
            self.assertTrue(self.check_equiv(self.aconv(x), conv(x)))
