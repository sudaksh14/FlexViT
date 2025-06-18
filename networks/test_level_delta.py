import unittest

from torch import nn
from adapt_modules.module import Module
import torch

from networks.adapt_model import AdaptModel

import networks.level_delta_utils as levels
import networks.resnet as resnet
import networks.resnetadapt as resneta

import utils


class TestLevelDelta(unittest.TestCase):
    @staticmethod
    def check_equiv(a, b, error=1e5):
        return (torch.abs(a - b) < error).all()

    def test_deltas(self):
        amodel = resneta.Resnet(resneta.ResnetConfig())
        model = resnet.Resnet(resnet.ResnetConfig())
        utils.flexible_model_copy(amodel, model)

        x = torch.rand(10, 3, 100, 100)
        self.assertTrue(self.check_equiv(amodel(x), model(x)))

        delta_manager = levels.InMemoryDeltaManager(
            *levels.get_model_deltas(amodel))

        for i in range(amodel.max_level() - 1, -1, -1):
            amodel.set_level_use(i)
            delta_manager.move_model_to(model, i + 1, i)
            x = torch.rand(10, 3, 100, 100)
            self.assertTrue(self.check_equiv(amodel(x), model(x)))

        for i in range(1, amodel.max_level() + 1):
            amodel.set_level_use(i)
            delta_manager.move_model_to(model, i - 1, i)
            x = torch.rand(10, 3, 100, 100)
            self.assertTrue(self.check_equiv(amodel(x), model(x)))
