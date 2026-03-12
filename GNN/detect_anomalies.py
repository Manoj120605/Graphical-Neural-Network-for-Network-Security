"""
detect_anomalies.py
===================
Detects anomalous nodes by measuring how far each node's embedding
deviates from the mean embedding of its neighbors (L2 distance).

Pipeline:
    1. Load the synthetic graph from  synthetic_graph.pt
    2. Generate embeddings with the GraphSAGE encoder
    3. For each node, compute L2(embedding, mean_neighbor_embedding)
    4. Rank nodes by anomaly score and flag outliers
"""

import torch
import numpy as np
from model import GraphSAGEEncoder


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
    adj = {i: [] for i in range(num_nodes)}
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for s, d in zip(src, dst):
        adj[s].append(d)

    for node in range(num_nodes):
        neighbors = adj[node]
        if len(neighbors) == 0:
            scores[node] = 0.0
            continue

        neighbor_embs = embeddings[neighbors]           # [k, D]
        neighbor_mean = neighbor_embs.mean(dim=0)       # [D]
        scores[node] = torch.norm(embeddings[node] - neighbor_mean, p=2)

    return scores


def main():
    # ── 1. Load graph ──
    data = torch.load("synthetic_graph.pt", weights_only=False)
    print(f"Graph: {data.num_nodes} nodes, {data.edge_index.shape[1]} directed edges\n")

    # ── 2. Generate embeddings ──
    model = GraphSAGEEncoder(in_dim=16, hidden_dim=32, out_dim=16)
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)

    # ── 3. Compute anomaly scores ──
    scores = compute_anomaly_scores(embeddings, data.edge_index)

    # ── 4. Report ──
    print(f"{'Node':>6}  {'Score':>10}  {'Label':>7}  {'Status'}")
    print("-" * 42)

    ranked = torch.argsort(scores, descending=True)
    for node in ranked:
        node_id = node.item()
        score = scores[node_id].item()
        label = data.y[node_id].item()
        tag = "⚠ ANOMALY" if label == 1 else ""
        print(f"{node_id:>6}  {score:>10.4f}  {label:>7}  {tag}")

    # ── 5. Threshold-based detection ──
    mean_score = scores.mean().item()
    std_score = scores.std().item()
    threshold = mean_score + 2 * std_score          # 2-sigma rule

    flagged = (scores > threshold).nonzero(as_tuple=True)[0].tolist()
    true_anomalies = data.y.nonzero(as_tuple=True)[0].tolist()

    print(f"\n{'─' * 42}")
    print(f"Mean score     : {mean_score:.4f}")
    print(f"Std  score     : {std_score:.4f}")
    print(f"Threshold (μ+2σ): {threshold:.4f}")
    print(f"\nFlagged nodes  : {flagged}")
    print(f"True anomalies : {true_anomalies}")

    hit = set(flagged) & set(true_anomalies)
    print(f"Detected       : {list(hit) if hit else 'None (model is untrained)'}")

    # ── 6. Save scores ──
    np.savetxt("anomaly_scores.csv",
               np.column_stack([np.arange(data.num_nodes),
                                scores.numpy(),
                                data.y.numpy()]),
               header="node_id,anomaly_score,true_label",
               delimiter=",", fmt=["%d", "%.6f", "%d"], comments="")
    print("\n✔ Saved anomaly_scores.csv")


if __name__ == "__main__":
    main()
