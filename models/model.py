"""
model.py
========
GraphSAGE encoder for node-level embedding generation.

Architecture
    SAGEConv(16 → 32) → ReLU → Dropout(0.3)
    SAGEConv(32 → 16) → node embeddings
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


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
        Forward pass — returns node embeddings.

        Args:
            x:          Node feature matrix  [N, in_dim]
            edge_index: COO edge index       [2, E]

        Returns:
            Embedding matrix  [N, out_dim]
        """
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h
