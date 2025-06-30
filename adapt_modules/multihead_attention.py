from torch import nn
import torch

from adapt_modules.module import Module
import torch.nn.functional as F

from typing import Iterable
import copy

from typing import Union, Iterable


class SelfAttention(Module):
    def __init__(self, token_size: Iterable[int], heads: Iterable[int], dropout=0.0):
        super().__init__()

        assert (all(i % j == 0 for i, j in zip(token_size, heads)))
        head_dims = [i // j for i, j in zip(token_size, heads)]
        assert (sorted(head_dims) == head_dims)

        self.token_size = token_size
        self.proj_token_size = token_size
        self.heads = heads
        self.dropout = dropout

        self.max_heads = heads[-1]
        self.max_token_size = token_size[-1]
        self.level = self.max_level()

        self.in_weights = nn.Parameter(
            torch.rand(3 * self.max_token_size, self.max_token_size))
        self.in_bias = nn.Parameter(torch.rand(3 * self.max_token_size))
        self.out_weights = nn.Parameter(
            torch.rand(self.max_token_size, self.max_token_size))
        self.out_bias = nn.Parameter(torch.rand(self.max_token_size))

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2)
        token_trim = self.max_token_size - self.token_size[self.level]

        x = F.pad(x, (0, token_trim))

        seq_length, batch_size, token_size = x.shape
        proj = F.linear(x, self.in_weights, self.in_bias)
        proj = proj.unflatten(-1, (3, token_size))
        proj = proj.unsqueeze(0)
        proj = proj.transpose(0, -2)
        proj = proj.squeeze(-2)
        proj = proj.contiguous()
        proj = proj.view(3, seq_length, batch_size, token_size)

        regular_head_dim = self.max_token_size // self.max_heads

        proj = proj.view(3, seq_length, batch_size *
                         self.max_heads, regular_head_dim)
        proj = proj.transpose(1, 2)
        proj = proj.view(3, batch_size, self.max_heads,
                         seq_length, regular_head_dim)

        proj[:, :, self.heads[self.level]:, :, :] = proj[
            :, :, self.heads[self.level]:, :, :].zero_()

        adapted_head_dim = self.token_size[self.level] // self.heads[self.level]
        proj[:, :, :self.heads[self.level], :, adapted_head_dim:] = proj[
            :, :, :self.heads[self.level], :, adapted_head_dim:].zero_()

        attn_output = F.scaled_dot_product_attention(
            *proj, None, self.dropout, False)
        attn_output = attn_output.permute(2, 0, 1, 3)
        attn_output = attn_output.contiguous()
        attn_output = attn_output.view(batch_size * seq_length, token_size)

        attn_output = F.linear(attn_output, self.out_weights, self.out_bias)
        attn_output = attn_output.view(
            seq_length, batch_size, attn_output.size(1))

        attn_output = attn_output[:, :, :token_size - token_trim]
        attn_output = attn_output.permute(1, 0, 2)
        return attn_output

    def set_level_use(self, level: int) -> None:
        assert (level >= 0)
        assert (level <= self.max_level())
        self.level = level

    def current_level(self) -> int:
        return self.level

    def max_level(self) -> int:
        return len(self.token_size) - 1

    @staticmethod
    def base_type() -> type[nn.MultiheadAttention]:
        return nn.MultiheadAttention

    def copy_to_base(self, dest: nn.MultiheadAttention) -> None:
        w_in = self.in_weights.view(
            3, self.max_token_size, self.max_token_size)
        w_in = w_in[
            :, :self.token_size[self.level], :self.token_size[self.level]]
        w_in = w_in.reshape(
            3 * self.token_size[self.level], self.token_size[self.level])
        dest.in_proj_weight.data = w_in

        b_in = self.in_bias.view(3, self.max_token_size)
        b_in = b_in[:, :self.token_size[self.level]]
        b_in = b_in.reshape(3 * self.token_size[self.level])
        dest.in_proj_bias.data = b_in

        dest.out_proj.weight.data[:] = self.out_weights.data[
            :self.token_size[self.level], :self.token_size[self.level]]
        dest.out_proj.bias.data = self.out_bias.data[:self.token_size[self.level]]

    def load_from_base(self, src: nn.MultiheadAttention) -> None:
        self.in_weights.data[:] = src.in_proj_weight.data
        self.in_bias.data[:] = src.in_proj_bias.data
        self.out_weights.data[:] = src.out_proj.weight.data
        self.out_bias.data[:] = src.out_proj.bias.data

    def make_base_copy(self) -> nn.MultiheadAttention:
        lin = nn.MultiheadAttention(
            self.hidden_dim[self.level], self.num_heads, *self._args, **self._kwargs)
        self.copy_to_base(lin)
        return lin
