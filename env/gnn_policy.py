# env/gnn_policy.py
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from typing import Tuple

class TrafficGATActor(nn.Module):
    """
    Graph Attention Network actor for traffic signal control.
    Input: graph of junction nodes, each with 70-float observation (69 + 1 priority).
    Output: per-node action logits for MultiDiscrete([8, 4]).
    """
    def __init__(
        self,
        node_feature_dim: int = 70,    
        edge_feature_dim: int = 4,     
        hidden_dim: int = 128,
        n_heads: int = 4,
        action_phase_dim: int = 8,
        action_duration_dim: int = 4,
    ):
        super().__init__()

        # Two-layer GAT
        self.gat1 = GATConv(
            in_channels=node_feature_dim,
            out_channels=hidden_dim,
            heads=n_heads,
            concat=True,
            edge_dim=edge_feature_dim,
            dropout=0.1,
        )
        self.gat2 = GATConv(
            in_channels=hidden_dim * n_heads,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            edge_dim=edge_feature_dim,
            dropout=0.1,
        )

        # Separate heads for phase and duration
        self.phase_head    = nn.Linear(hidden_dim, action_phase_dim)     # logits over 8 phases
        self.duration_head = nn.Linear(hidden_dim, action_duration_dim)  # logits over 4 durations

        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_features: torch.Tensor,   # (n_nodes, 70)
        edge_index: torch.Tensor,      # (2, n_edges)
        edge_attr: torch.Tensor,       # (n_edges, 4)
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.activation(self.gat1(node_features, edge_index, edge_attr))
        x = self.layer_norm(self.activation(self.gat2(x, edge_index, edge_attr)))

        phase_logits    = self.phase_head(x)      # (n_nodes, 8)
        duration_logits = self.duration_head(x)   # (n_nodes, 4)

        return phase_logits, duration_logits

class TrafficGATCritic(nn.Module):
    """
    Centralised critic for MAPPO.
    Aggregates ALL node features via global mean pool → single value.
    """
    def __init__(self, node_feature_dim: int = 70, hidden_dim: int = 256, n_heads: int = 4):
        super().__init__()
        self.gat1 = GATConv(node_feature_dim, hidden_dim, heads=n_heads, concat=True)
        self.gat2 = GATConv(hidden_dim * n_heads, hidden_dim, heads=1, concat=False)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, node_features, edge_index, edge_attr, batch=None):
        x = torch.relu(self.gat1(node_features, edge_index, edge_attr))
        x = torch.relu(self.gat2(x, edge_index, edge_attr))
        x_pooled = global_mean_pool(x, batch) if batch is not None else x.mean(dim=0, keepdim=True)
        return self.value_head(x_pooled)