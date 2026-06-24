# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class FeatureTokenizer(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, embed_dim))
        self.bias = nn.Parameter(torch.zeros(num_features, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = x.unsqueeze(-1)
        return x * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

class MSFFN(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.branch2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.Dropout(dropout)
        )
        self.merge = nn.Linear(embed_dim * 4, embed_dim)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        return self.merge(torch.cat([out1, out2], dim=-1))

class MSFFNTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=256,
                 dropout=0.1, activation="gelu", batch_first=True):
        super().__init__(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation,
            batch_first=batch_first
        )
        self.ms_ffn = MSFFN(d_model, dropout=dropout)
        self.linear1 = nn.Identity()
        self.linear2 = nn.Identity()

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask, need_weights=False)[0]
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.ms_ffn(src)
        return self.norm2(src + self.dropout2(src2))

class FTTransformerMulti(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(input_dim, embed_dim)

        encoder_layer = MSFFNTransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)

        self.shared = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.head_perf = nn.Linear(64, 2)

    def forward(self, x, return_feat: bool = False):
        """
        Args:
            x: [B, F]
            return_feat: 是否返回中间特征
        Returns:
            if return_feat:
                feat: [B, 64], pred: [B, 2]
            else:
                pred: [B, 2]
        """
        tokens = self.tokenizer(x)                 # [B, F, D]
        encoded = self.encoder(tokens)             # [B, F, D]
        pooled = self.norm(encoded.mean(dim=1))    # [B, D]
        feat = self.shared(pooled)                 # [B, 64]
        pred = self.head_perf(feat)                # [B, 2]

        if return_feat:
            return feat, pred
        else:
            return pred

