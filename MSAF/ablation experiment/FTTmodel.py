# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


# =========================================================
# Feature Tokenizer
# =========================================================
class FeatureTokenizer(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, embed_dim))
        self.bias = nn.Parameter(torch.zeros(num_features, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = x.unsqueeze(-1)
        return x * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


# =========================================================
# Standard Transformer Encoder Layer
# =========================================================
class VanillaTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True
        )


        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src):
        # Self Attention
        attn_out, _ = self.self_attn(src, src, src)
        src = self.norm1(src + attn_out)

        # FFN
        ffn_out = self.ffn(src)
        src = self.norm2(src + ffn_out)

        return src


# =========================================================
# FT-Transformer
# =========================================================
class FTTransformerMulti(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()

        self.tokenizer = FeatureTokenizer(input_dim, embed_dim)

        self.layers = nn.ModuleList([
            VanillaTransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        
        self.shared = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.head_perf = nn.Linear(64, 2)

    def forward(self, x, return_feat: bool = False):
        tokens = self.tokenizer(x)  # [B, F, D]

        for layer in self.layers:
            tokens = layer(tokens)

        pooled = self.norm(tokens.mean(dim=1))
        feat = self.shared(pooled)
        pred = self.head_perf(feat)

        if return_feat:
            return feat, pred
        else:
            return pred