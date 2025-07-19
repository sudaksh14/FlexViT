import torch
from torch import nn

from typing import Callable, Iterable
from functools import partial

import math

from collections import OrderedDict
import utils

import dataclasses
from networks.config import ModelConfig
# from networks.adapt_model import AdaptModel

from enum import Enum
from torchvision.models import vision_transformer

import copy
from networks.flex_model import FlexModel

from networks.vit import ViTStructureConfig, ViTStructure, ViTPrebuilt, KNOWN_MODEL_PRETRAINED, DEFAULT_NUM_CLASSES
import networks.vit

from flex_modules.pos_embedding import PosEmbeddingLayer
from flex_modules.class_token import ClassTokenLayer
from flex_modules.linear import Linear
from flex_modules.conv2d import Conv2d
from flex_modules.layer_norm import LayerNorm
from flex_modules.selfattention import SelfAttention
import flex_modules as am


# This model is mostly an adapted version from torchvision.models.vision_transformer

def scale_with_heads_list(heads, max_hidden_dims):
    max_num_heads = heads[-1]
    if max_hidden_dims % max_num_heads:
        raise RuntimeError()
    return [max_hidden_dims // max_num_heads * i for i in heads]


@utils.fluent_setters
@dataclasses.dataclass
class ViTConfig(ModelConfig):
    structure: ViTStructureConfig = ViTStructure.b16
    prebuilt: ViTPrebuilt = ViTPrebuilt.default
    num_classes: int = DEFAULT_NUM_CLASSES
    dropout: float = 0.0
    attention_dropout: float = 0.0

    hidden_dims: Iterable[int] = ((768 // 3) * 2, 768)
    num_heads: Iterable[int] = (8, 12)
    mlp_dims: Iterable[int] = ((3072 // 3) * 2, 3072)

    def make_model(self):
        return VisionTransformer(self)

    def no_prebuilt(self):
        self.prebuilt = ViTPrebuilt.noprebuild
        return self

    def create_base_config(self, level) -> ModelConfig:
        return networks.vit.ViTConfig(
            networks.vit.ViTStructureConfig(
                self.structure.image_size,
                self.structure.patch_size,
                self.structure.num_layers,
                self.num_heads[level],
                self.hidden_dims[level],
                self.mlp_dims[level]),
            self.prebuilt,
            self.num_classes,
            self.dropout,
            self.attention_dropout
        )


class MLPBlock(nn.Sequential):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: Iterable[int], mlp_dim: Iterable[int], dropout: float):
        super().__init__(
            Linear(in_dim, mlp_dim, bias=True),
            nn.GELU(),
            torch.nn.Dropout(dropout),
            Linear(mlp_dim, in_dim, bias=True),
            torch.nn.Dropout(dropout),
        )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: Iterable[int],
        mlp_dim: Iterable[int],
        dropout: float,
        attention_dropout: float
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = LayerNorm(hidden_dim, eps=1e-6)
        self.self_attention = SelfAttention(
            hidden_dim, num_heads, dropout=attention_dropout)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim(
        ) == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x = self.self_attention(x)

        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)

        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: Iterable[int],
        mlp_dim: Iterable[int],
        dropout: float,
        attention_dropout: float
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = PosEmbeddingLayer(seq_length, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout
            )
        self.layers = nn.Sequential(layers)
        self.ln = LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim(
        ) == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = self.pos_embedding(input)
        return self.ln(self.layers(self.dropout(input)))


class VisionTransformer(FlexModel):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        config: ViTConfig,
    ):
        image_size = config.structure.image_size
        patch_size = config.structure.patch_size
        num_layers = config.structure.num_layers
        num_heads = config.num_heads
        hidden_dim = config.hidden_dims
        mlp_dim = config.mlp_dims

        num_classes = config.num_classes
        dropout = config.dropout
        attention_dropout = config.attention_dropout

        super().__init__()
        torch._assert(image_size % patch_size == 0,
                      "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes

        self.class_token = ClassTokenLayer(hidden_dim)

        self.conv_proj = Conv2d(
            [3] * len(hidden_dim), hidden_dim, kernel_size=patch_size, stride=patch_size)

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout
        )

        self.seq_length = seq_length

        self.heads = am.LinearSelect(
            hidden_dim, [DEFAULT_NUM_CLASSES] * len(hidden_dim))

        self.set_level_use(self.max_level())
        self.level = self.max_level()
        if config.prebuilt != ViTPrebuilt.noprebuild:
            prebuild_config = (config.structure, config.prebuilt)
            if prebuild_config not in KNOWN_MODEL_PRETRAINED:
                raise RuntimeError("prebuilt model not found")
            prebuilt = KNOWN_MODEL_PRETRAINED[prebuild_config]()
            utils.flexible_model_copy(prebuilt, self)
            self.class_token.token = copy.deepcopy(prebuilt.class_token)
            self.encoder.pos_embedding.embedding = copy.deepcopy(
                prebuilt.encoder.pos_embedding)

        if config.num_classes != DEFAULT_NUM_CLASSES:
            self.heads = am.LinearSelect(
                hidden_dim, [num_classes] * len(hidden_dim))

    @staticmethod
    def base_type() -> type[nn.Module]:
        return networks.vit.VisionTransformer

    def current_level(self) -> int:
        return self.level

    def max_level(self) -> int:
        return len(self.hidden_dim) - 1

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size,
                      f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size,
                      f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim[self.current_level()], n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        x = self.class_token(x, n)
        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x

    @torch.no_grad()
    def export_level_delta(self) -> tuple[int, int]:
        return self.hidden_dim[self.level], self.hidden_dim[self.level]

    @staticmethod
    def apply_level_delta_down(model: networks.vit.VisionTransformer, level_delta: int) -> None:
        model.hidden_dim = level_delta

    @staticmethod
    def apply_level_delta_up(model: networks.vit.VisionTransformer, level_delta: int) -> None:
        model.hidden_dim = level_delta
