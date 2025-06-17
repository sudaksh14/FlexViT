from torch import nn
import torch

from adapt_modules.module import Module
from adapt_modules.adapt_select import AdaptSelect


class BatchNorm2d(AdaptSelect):
    def __init__(self, channels, *args, **kwargs):
        layers = [
            nn.BatchNorm2d(c, *args, **kwargs) for c in channels
        ]
        super().__init__(layers)

    def base_type(self):
        return nn.BatchNorm2d
