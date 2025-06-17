from torch import nn
import torch

from adapt_modules.module import Module
import torch.nn.functional as F


class Conv2d(Module):
    def __init__(self, in_sizes, out_sizes, *args, **kwargs):
        super().__init__()
        self.in_sizes = in_sizes
        self.out_sizes = out_sizes

        assert (not kwargs.get('bias', False))
        assert (len(self.in_sizes) > 0)
        assert (len(self.out_sizes) > 0)
        assert (len(self.in_sizes) == len(self.out_sizes))
        assert (sorted(self.in_sizes) == list(self.in_sizes))
        assert (sorted(self.out_sizes) == list(self.out_sizes))
        assert (self.in_sizes[0] > 0)
        assert (self.out_sizes[0] > 0)

        self.max_in_size = self.in_sizes[-1]
        self.max_out_size = self.out_sizes[-1]

        self._cached = (-1, -1, -1)
        self._zeros_cache = None

        self.set_level_use(self.max_level())
        kwargs['bias'] = False
        self.conv = nn.Conv2d(
            self.max_in_size, self.max_out_size, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 0, 0, 0, 0,
                      self.max_in_size - self.in_sizes[self.level]))
        x = self.conv(x)
        x = x[:, :self.out_sizes[self.level]]
        return x

    def set_level_use(self, level: int):
        assert (level >= 0)
        assert (level <= self.max_level())
        self.level = level

    def current_level(self) -> int:
        return self.level

    def max_level(self) -> int:
        return len(self.in_sizes)

    def base_type(self):
        return nn.Conv2d

    def copy_to_base(self, dest: nn.Conv2d):
        dest.weight.data = self.conv.weight.data[:self.out_sizes[self.level],
                                                 :self.in_sizes[self.level]]

    def load_from_base(self, src: nn.Conv2d):
        self.conv.weight.data[:self.out_sizes[self.level],
                              :self.in_sizes[self.level]] = src.weight.data
