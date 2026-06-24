# -*- coding: utf-8 -*-
"""
Raft-GAT (Star Topology, Condition-aware, Stable)

- Leader-centered Raft graph
- Role + system-condition node features
- Learnable structural gate
- Designed as structural regularizer for Hybrid FTT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================
# Single GAT Layer
# ======================================================
class RaftGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(out_dim * 2, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h, adj):
        """
        h   : [B, N, F]
        adj : [B, N, N]
        """
        B, N, _ = h.shape

        Wh = self.fc(h)  # [B, N, out]

        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)

        e = self.leaky_relu(
            self.attn(torch.cat([Wh_i, Wh_j], dim=-1))
        ).squeeze(-1)  # [B, N, N]

        e = e.masked_fill(adj == 0, float("-inf"))
        alpha = F.softmax(e, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)

        return torch.matmul(alpha, Wh)


# ======================================================
# RaftGAT Network
# ======================================================
class RaftGAT(nn.Module):
    """
    Condition-aware Raft Graph Attention Network
    """

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

        # learnable but bounded structural gate
        self.gamma = nn.Parameter(torch.tensor(init_scale))

        self.layers = nn.ModuleList()
        self.layers.append(RaftGATLayer(node_dim, hidden))
        for _ in range(layers - 1):
            self.layers.append(RaftGATLayer(hidden, hidden))

        self.readout = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.ReLU()
        )

    # ==================================================
    # Build Star Raft Adjacency 
    # ==================================================
    def build_star_adj(self, topo, device):
        """
        topo : [B, 1]  number of orderers
        """
        B = topo.size(0)
        N = self.max_orderers

        adj = torch.zeros(B, N, N, device=device)

        # leader <-> followers
        adj[:, 0, 1:] = 1.0
        adj[:, 1:, 0] = 1.0

        # self-loop
        adj += torch.eye(N, device=device).unsqueeze(0)

        # mask invalid nodes
        for b in range(B):
            valid = int(topo[b].item())
            if valid < N:
                adj[b, valid:, :] = 0
                adj[b, :, valid:] = 0

        return adj

    # ==================================================
    # Node Features: role + system conditions
    # ==================================================
    def build_node_features(self, x, topo):
        """
        x    : [B, F]  raw input (arrival, orderers, block, ...)
        topo : [B, 1]
        """
        B = x.size(0)
        device = x.device
        N = self.max_orderers

        h = torch.zeros(B, N, 4, device=device)

        # role
        h[:, 0, 0] = 1.0  # leader

        # system conditions (broadcast)
        h[:, :, 1] = x[:, 0].unsqueeze(1)  # arrival
        h[:, :, 2] = x[:, 2].unsqueeze(1)  # block size
        h[:, :, 3] = topo.float()           # orderers

        return h

    # ==================================================
    # Forward
    # ==================================================
    def forward(self, x, topo=None):
        """
        x    : [B, F]
        topo : [B, 1]
        """
        B = x.size(0)
        device = x.device

        if topo is None:
            topo = torch.full(
                (B, 1),
                self.max_orderers,
                device=device,
                dtype=torch.long
            )

        adj = self.build_star_adj(topo, device)
        h = self.build_node_features(x, topo)

        # bounded structural contribution
        h = h * torch.clamp(self.gamma, 0.01, 0.3)

        for layer in self.layers:
            h = layer(h, adj)

        h = h.mean(dim=1)
        return self.readout(h)
