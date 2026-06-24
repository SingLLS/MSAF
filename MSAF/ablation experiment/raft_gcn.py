# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class RaftGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, h, adj):
        """
        h   : [B, N, F]
        adj : [B, N, N]
        """
        deg = adj.sum(dim=-1, keepdim=True) + 1e-6
        h_agg = torch.matmul(adj, h) / deg
        return self.fc(h_agg)


# ======================================================
# Raft GCN Network
# ======================================================
class RaftGCN(nn.Module):
    """
    Raft Graph Convolution Network (No Attention)
    """

    def __init__(
        self,
        node_dim=4,
        hidden=32,
        out_dim=64,
        layers=2,
        max_orderers=7,
        init_scale=0.1
    ):
        super().__init__()

        self.max_orderers = max_orderers
        self.gamma = nn.Parameter(torch.tensor(init_scale))

        self.layers = nn.ModuleList()
        self.layers.append(RaftGCNLayer(node_dim, hidden))
        for _ in range(layers - 1):
            self.layers.append(RaftGCNLayer(hidden, hidden))

        self.readout = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.ReLU()
        )

    def build_star_adj(self, topo, device):
        B = topo.size(0)
        N = self.max_orderers

        adj = torch.zeros(B, N, N, device=device)

        adj[:, 0, 1:] = 1.0
        adj[:, 1:, 0] = 1.0
        adj += torch.eye(N, device=device).unsqueeze(0)

        for b in range(B):
            valid = int(topo[b].item())
            if valid < N:
                adj[b, valid:, :] = 0
                adj[b, :, valid:] = 0

        return adj

    def build_node_features(self, x, topo):
        B = x.size(0)
        device = x.device
        N = self.max_orderers

        h = torch.zeros(B, N, 4, device=device)

        h[:, 0, 0] = 1.0
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

        adj = self.build_star_adj(topo, device)
        h = self.build_node_features(x, topo)

        h = h * torch.clamp(self.gamma, 0.01, 0.3)

        for layer in self.layers:
            h = layer(h, adj)

        h = h.mean(dim=1)
        return self.readout(h)
