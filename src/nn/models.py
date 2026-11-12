"""Arquitecturas neuronales para datos tabulares (PyTorch).

1. ``MLPNet``        — Perceptrón multicapa con *embeddings* para las variables
                       nominales (programa, ocupación de los padres, etc.).
2. ``FTTransformer`` — *Feature Tokenizer + Transformer* (Gorishniy et al.,
                       2021): cada variable se convierte en un *token* y un
                       codificador Transformer (auto-atención) los combina; la
                       salida se lee del token [CLS].
"""

from __future__ import annotations

import math

import torch
from torch import nn


class MLPNet(nn.Module):
    def __init__(self, n_num: int, cardinalities: list[int], n_classes: int = 3,
                 emb_dim: int = 4, hidden: tuple[int, ...] | list[int] = (64,), dropout: float = 0.3):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, emb_dim) for c in cardinalities])
        d_in = n_num + emb_dim * len(cardinalities)
        layers: list[nn.Module] = []
        for h in hidden:
            layers += [nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(dropout)]
            d_in = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d_in, n_classes)   # logits; softmax en la pérdida / inferencia

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        parts = [x_num] + [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        return self.head(self.body(torch.cat(parts, dim=1)))


class FTTransformer(nn.Module):
    def __init__(self, n_num: int, cardinalities: list[int], n_classes: int = 3,
                 d_token: int = 32, n_layers: int = 2, n_heads: int = 4,
                 ffn_mult: int = 2, dropout: float = 0.2):
        super().__init__()
        assert d_token % n_heads == 0, "d_token debe ser múltiplo de n_heads"
        bound = 1 / math.sqrt(d_token)
        # Tokenizador numérico: token_j = x_j * w_j + b_j
        self.num_w = nn.Parameter(torch.empty(n_num, d_token).uniform_(-bound, bound))
        self.num_b = nn.Parameter(torch.empty(n_num, d_token).uniform_(-bound, bound))
        self.cat_embs = nn.ModuleList([nn.Embedding(c, d_token) for c in cardinalities])
        self.cls = nn.Parameter(torch.empty(1, 1, d_token).uniform_(-bound, bound))
        layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * ffn_mult,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, n_classes)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        b = x_num.shape[0]
        tok_num = x_num.unsqueeze(-1) * self.num_w + self.num_b                       # (B, n_num, d)
        tok_cat = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs)], dim=1)  # (B, n_cat, d)
        tokens = torch.cat([self.cls.expand(b, -1, -1), tok_num, tok_cat], dim=1)
        h = self.encoder(tokens)[:, 0]                                                # token [CLS]
        return self.head(self.norm(h))


ARCHS = {"mlp": MLPNet, "ft_transformer": FTTransformer}
ARCH_NAMES = {"mlp": "Red neuronal MLP (embeddings)", "ft_transformer": "FT-Transformer (auto-atención)"}


def build_model(arch: str, n_num: int, cardinalities: list[int], n_classes: int, params: dict) -> nn.Module:
    return ARCHS[arch](n_num, cardinalities, n_classes, **params)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
