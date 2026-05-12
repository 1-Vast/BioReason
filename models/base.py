"""Basic building blocks: MLP, ResidualBlock, EmbeddingBlock."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple multi-layer perceptron.

    [B, in_dim] -> [B, out_dim]
    """

    def __init__(self, in_dim, out_dim, hidden=None, layers=2, dropout=0.1, act=nn.GELU, norm=True):
        super().__init__()
        hidden = hidden or out_dim
        dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        net = []
        for i in range(len(dims) - 1):
            net.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if norm:
                    net.append(nn.LayerNorm(dims[i + 1]))
                net.append(act())
                net.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm + Linear + Act + Dropout + Linear."""

    def __init__(self, dim, hidden=None, dropout=0.1, act=nn.GELU):
        super().__init__()
        hidden = hidden or dim * 4
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = act()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return residual + x


class EmbeddingBlock(nn.Module):
    """Categorical embedding lookup with optional projection."""

    def __init__(self, num_embeddings, dim, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.embed(x))
