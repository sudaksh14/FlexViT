import unittest
import sys

from torch import nn
import torch

from flex_modules.module import Module
import flex_modules as fm
import utils

from networks import flexvgg, flexvit, flexresnet, vgg, vit, resnet


class SelfAttention(nn.MultiheadAttention):
    def forward(self, x):
        return super().forward(x, x, x)[0]


x = torch.rand(1, 10, 100)
model = SelfAttention(100, 5, batch_first=True)

y = model(x)

print(y)
