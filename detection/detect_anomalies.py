"""
detect_anomalies.py
===================
Detects anomalous nodes by measuring how far each node's embedding
deviates from the mean embedding of its neighbors (L2 distance).

Threshold : mean + 2*std  (2-sigma rule, SYSTEM_DESIGN Section 8)
Alert     : writes a JSON alert file for every flagged node.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import torch
from torch_geometric.data import Data

from models import GraphSAGEEncoder
from models.model import get_device


def compute_anomaly_scores(embeddings: torch.Tensor,
                           edge_index: torch.Tensor) -> torch.Tensor:
    """
    For every node compute L2(embedding, mean_neighbor_embedding).

    Args:
        embeddings: [N, D] node embedding matrix
        edge_index: [2, E] COO edge index (undirected)

    Returns:
        scores: [N] anomaly score per node (higher = more anomalous)
    """
    # Move to CPU for scoring (lightweight, avoids device issues)
    embeddings = embeddings.cpu()
    edge_index = edge_index.cpu()

    num_nodes = embeddings.size(0)
    scores = torch.zeros(num_nodes)

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


def _generate_alerts(flagged: list[int],
                     scores: torch.Tensor,
                     threshold: float,
                     output_dir: str) -> str:
    """
    Write a JSON alert file for flagged nodes.
    Each alert contains node id, score, threshold, and timestamp.
    """
    alerts = []
    for nid in flagged:
        alerts.append({
            "alert_type": "ANOMALY_DETECTED",
            "node_id": nid,
            "anomaly_score": round(scores[nid].item(), 6),
            "threshold": round(threshold, 6),
            "severity": "HIGH" if scores[nid].item() > threshold * 1.5 else "MEDIUM",
            "reason": "deviation from neighborhood embedding",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    alert_path = os.path.join(output_dir, "alerts.json")
    with open(alert_path, "w", encoding="utf-8") as f:
        json.dump({"alerts": alerts, "total": len(alerts)}, f, indent=2)

    return alert_path


def detect(data: Data,
           output_dir: str = "syntheticdata") -> tuple[torch.Tensor, list[int], float]:
    """
    End-to-end anomaly detection:
        1. Move data to device (GPU if available)
        2. Embed nodes with GraphSAGE
        3. Score each node (L2 neighbor deviation)
        4. Flag nodes above mean + 2*std
        5. Save anomaly_scores.csv and alerts.json

    Returns:
        (scores, flagged_node_ids, threshold)
    """
    os.makedirs(output_dir, exist_ok=True)

    # -- Device selection (SYSTEM_DESIGN Section 12) --
    device = get_device()
    print("[detection] Device : %s" % device)

    model = GraphSAGEEncoder(in_dim=16, hidden_dim=32, out_dim=16).to(device)
    model.eval()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    with torch.no_grad():
        embeddings = model(x, edge_index)

    # -- Score (back on CPU) --
    scores = compute_anomaly_scores(embeddings, edge_index)

    # -- Threshold (2-sigma rule) --
    mean_score = scores.mean().item()
    std_score = scores.std().item()
    threshold = mean_score + 2 * std_score

    flagged = (scores > threshold).nonzero(as_tuple=True)[0].tolist()
    true_anomalies = data.y.nonzero(as_tuple=True)[0].tolist()
    detected = sorted(set(flagged) & set(true_anomalies))

    # -- Report --
    print("")
    print("[detection] Scores  : mu=%.4f  sigma=%.4f" % (mean_score, std_score))
    print("[detection] Threshold (mu+2s) : %.4f" % threshold)
    print("[detection] Flagged : %s" % flagged)
    print("[detection] True    : %s" % true_anomalies)
    print("[detection] Hit     : %s" % (detected if detected else "None (model untrained)"))

    # -- Per-node table --
    ranked = torch.argsort(scores, descending=True)
    print("")
    print("%6s  %10s  %7s  %s" % ("Node", "Score", "Label", "Status"))
    print("-" * 45)
    for node in ranked:
        nid = node.item()
        s = scores[nid].item()
        lbl = data.y[nid].item()
        if lbl == 1:
            tag = "<< ANOMALY"
        elif nid in flagged:
            tag = "<< FLAGGED"
        else:
            tag = ""
        print("%6d  %10.4f  %7d  %s" % (nid, s, lbl, tag))

    # -- Save CSV --
    csv_path = os.path.join(output_dir, "anomaly_scores.csv")
    np.savetxt(csv_path,
               np.column_stack([np.arange(data.num_nodes),
                                scores.numpy(),
                                data.y.numpy()]),
               header="node_id,anomaly_score,true_label",
               delimiter=",", fmt=["%d", "%.6f", "%d"], comments="")
    print("")
    print("[detection] Saved scores : %s" % csv_path)

    # -- Save alerts (SYSTEM_DESIGN "Visualization + Alert") --
    alert_path = _generate_alerts(flagged, scores, threshold, output_dir)
    print("[detection] Saved alerts : %s" % alert_path)

    return scores, flagged, threshold


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    data = torch.load("syntheticdata/synthetic_graph.pt", weights_only=False)
    detect(data)
