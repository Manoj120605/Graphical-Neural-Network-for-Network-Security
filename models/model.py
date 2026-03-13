"""
model.py
========
GraphSAGE encoder for node-level embedding generation.

Architecture
    SAGEConv(16 -> 32) -> ReLU -> Dropout(0.3)
    SAGEConv(32 -> 16) -> node embeddings

Supports automatic CUDA device selection when a GPU is available.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class GraphSAGEEncoder(torch.nn.Module):
    """
    Two-layer GraphSAGE model that produces a low-dimensional
    embedding for every node in the graph.
    """

    def __init__(self, in_dim: int = 16, hidden_dim: int = 32,
                 out_dim: int = 16, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass -- returns node embeddings [N, out_dim].
        """
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h
