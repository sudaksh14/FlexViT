from torch import nn
import torch


class ClassTokenLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(self, x, n):
        batch_class_token = self.token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        return x


class PosEmbeddingLayer(nn.Module):
    def __init__(self, seq_length, hidden_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(
            1, seq_length, hidden_dim).normal_(std=0.02))

    def forward(self, x):
        return x + self.embedding
