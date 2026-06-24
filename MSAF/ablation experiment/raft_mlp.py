# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class RaftMLP(nn.Module):

    def __init__(
        self,
        node_dim=4,
        hidden=32,
        out_dim=64,
        layers=2,
        max_orderers=9,
        init_scale=0.1
    ):
        super().__init__()

        self.max_orderers = max_orderers
        self.gamma = nn.Parameter(torch.tensor(init_scale))

        mlp_layers = []
        in_dim = node_dim * max_orderers  # flatten all nodes

        mlp_layers.append(nn.Linear(in_dim, hidden))
        mlp_layers.append(nn.ReLU())

        for _ in range(layers - 1):
            mlp_layers.append(nn.Linear(hidden, hidden))
            mlp_layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*mlp_layers)

        self.readout = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.ReLU()
        )

    def build_node_features(self, x, topo):
        """
        Same node features as RaftGAT (for fairness)
        """
        B = x.size(0)
        device = x.device
        N = self.max_orderers

        h = torch.zeros(B, N, 4, device=device)

        h[:, 0, 0] = 1.0  # leader
        h[:, :, 1] = x[:, 0].unsqueeze(1)
        h[:, :, 2] = x[:, 2].unsqueeze(1)
        h[:, :, 3] = topo.float()

        return h

    def forward(self, x, topo=None):
        B = x.size(0)
        device = x.device

        if topo is None:
            topo = torch.full(
                (B, 1),
                self.max_orderers,
                device=device
            )

        h = self.build_node_features(x, topo)
        h = h * torch.clamp(self.gamma, 0.01, 0.3)

        h = h.view(B, -1)  # flatten nodes
        h = self.encoder(h)

        return self.readout(h)
