from torch import nn
import torch

from adapt_modules.module import Module
from adapt_modules.adapt_select import AdaptSelect


class Linear(AdaptSelect):
    def __init__(self, in_sizes, out_sizes, *args, **kwargs):
        layers = [
            nn.Linear(ins, outs, *args, **kwargs) for ins, outs in zip(in_sizes, out_sizes)
        ]
        super().__init__(layers)

    def base_type(self):
        return nn.Linear
