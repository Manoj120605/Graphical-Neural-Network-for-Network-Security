"""
detect_anomalies.py
===================
Detects anomalous nodes by measuring how far each node's embedding
deviates from the mean embedding of its neighbors (L2 distance).

Threshold: μ + 2σ  (2-sigma rule, per SYSTEM_DESIGN §8)
"""

import os
import torch
import numpy as np
from torch_geometric.data import Data

from models import GraphSAGEEncoder


def compute_anomaly_scores(embeddings: torch.Tensor,
                           edge_index: torch.Tensor) -> torch.Tensor:
    """
    For every node, compute the L2 distance between its embedding
    and the mean embedding of its 1-hop neighbors.

    Args:
        embeddings: [N, D] node embedding matrix
        edge_index: [2, E] COO edge index (undirected)

    Returns:
        scores: [N] anomaly score per node (higher = more anomalous)
    """
    num_nodes = embeddings.size(0)
    scores = torch.zeros(num_nodes)

    # Build adjacency list from edge_index
    adj: dict[int, list[int]] = {i: [] for i in range(num_nodes)}
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for s, d in zip(src, dst):
        adj[s].append(d)

    for node in range(num_nodes):
        neighbors = adj[node]
        if len(neighbors) == 0:
            scores[node] = 0.0
            continue
        neighbor_embs = embeddings[neighbors]
        neighbor_mean = neighbor_embs.mean(dim=0)
        scores[node] = torch.norm(embeddings[node] - neighbor_mean, p=2)

    return scores


def detect(data: Data,
           output_dir: str = "syntheticdata") -> tuple[torch.Tensor, list[int], float]:
    """
    End-to-end anomaly detection:
        1. Embed nodes with GraphSAGE
        2. Score each node
        3. Flag nodes above μ+2σ
        4. Save anomaly_scores.csv

    Returns:
        (scores, flagged_node_ids, threshold)
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Embed ──
    model = GraphSAGEEncoder(in_dim=16, hidden_dim=32, out_dim=16)
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)

    # ── Score ──
    scores = compute_anomaly_scores(embeddings, data.edge_index)

    # ── Threshold (2-sigma rule) ──
    mean_score = scores.mean().item()
    std_score = scores.std().item()
    threshold = mean_score + 2 * std_score

    flagged = (scores > threshold).nonzero(as_tuple=True)[0].tolist()
    true_anomalies = data.y.nonzero(as_tuple=True)[0].tolist()
    detected = sorted(set(flagged) & set(true_anomalies))

    # ── Report ──
    print(f"\n[detection] Scores  : μ={mean_score:.4f}  σ={std_score:.4f}")
    print(f"[detection] Threshold (μ+2σ) : {threshold:.4f}")
    print(f"[detection] Flagged : {flagged}")
    print(f"[detection] True    : {true_anomalies}")
    print(f"[detection] Hit     : {detected if detected else 'None (model untrained)'}")

    # ── Per-node table ──
    ranked = torch.argsort(scores, descending=True)
    print(f"\n{'Node':>6}  {'Score':>10}  {'Label':>7}  Status")
    print("-" * 45)
    for node in ranked:
        nid = node.item()
        s = scores[nid].item()
        lbl = data.y[nid].item()
        tag = "⚠ ANOMALY" if lbl == 1 else ("⚡ FLAGGED" if nid in flagged else "")
        print(f"{nid:>6}  {s:>10.4f}  {lbl:>7}  {tag}")

    # ── Save CSV ──
    csv_path = os.path.join(output_dir, "anomaly_scores.csv")
    np.savetxt(csv_path,
               np.column_stack([np.arange(data.num_nodes),
                                scores.numpy(),
                                data.y.numpy()]),
               header="node_id,anomaly_score,true_label",
               delimiter=",", fmt=["%d", "%.6f", "%d"], comments="")
    print(f"\n[detection] Saved → {csv_path}")

    return scores, flagged, threshold


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    data = torch.load("syntheticdata/synthetic_graph.pt", weights_only=False)
    detect(data)
