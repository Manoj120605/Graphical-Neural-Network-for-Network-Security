"""
model.py
========
GraphSAGE encoder for node-level embedding generation.

Architecture:
    SAGEConv(16 → 32) → ReLU → Dropout
    SAGEConv(32 → 16) → node embeddings

Usage:
    from model import GraphSAGEEncoder
    model = GraphSAGEEncoder(in_dim=16, hidden_dim=32, out_dim=16)
    embeddings = model(data.x, data.edge_index)   # shape [N, 16]
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
            Embedding matrix [N, out_dim]
        """
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)          # [N, out_dim]
        return h


# ────────────────────── Quick standalone test ──────────────────────────
if __name__ == "__main__":
    data = torch.load("synthetic_graph.pt", weights_only=False)
    print(f"Loaded graph: {data}")

    model = GraphSAGEEncoder(in_dim=16, hidden_dim=32, out_dim=16)
    model.eval()

    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)

    print(f"\nEmbedding shape: {embeddings.shape}")        # [50, 16]
    print(f"Anomaly node embedding:\n  {embeddings[0]}")
    print(f"Normal  node embedding:\n  {embeddings[1]}")

    # L2 distance between anomaly and a normal node
    dist = torch.norm(embeddings[0] - embeddings[1]).item()
    print(f"\nL2 distance (anomaly ↔ normal): {dist:.4f}")
