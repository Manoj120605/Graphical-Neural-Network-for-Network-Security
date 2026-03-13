"""
dual_plane_verify.py
====================
Dual-Plane Structural Poisoning Defense for AutoNet-GNN.

Implements two independent verification planes:
  1. FEATURE PLANE  — detects anomalous node feature vectors (L2 deviation)
  2. STRUCTURE PLANE — detects suspicious topology changes (degree anomalies,
                       edge density, neighbor correlation)

Cross-referencing both planes classifies each node as:
  - CLEAN             — normal on both planes
  - FEATURE_POISONED  — anomalous features, normal structure
  - STRUCTURE_POISONED — normal features, anomalous structure
  - DUAL_POISONED     — anomalous on BOTH planes (hardest to detect)
"""

import os
import json
import numpy as np
import torch
from datetime import datetime, timezone
from torch_geometric.data import Data

FEATURE_NAMES = [
    "traffic_in", "traffic_out", "packet_loss", "latency",
    "crc_errors", "cpu_usage", "memory_usage", "connection_count",
    "interface_errors", "dropped_packets", "jitter", "link_utilization",
    "route_changes", "neighbor_count", "retransmissions", "queue_depth",
]

CLASSIFICATIONS = {
    (False, False): "CLEAN",
    (True,  False): "FEATURE_POISONED",
    (False, True):  "STRUCTURE_POISONED",
    (True,  True):  "DUAL_POISONED",
}


# ═══════════════════════════════════════════════════════════════════════
# FEATURE PLANE
# ═══════════════════════════════════════════════════════════════════════

def verify_feature_plane(data: Data,
                         scores: torch.Tensor,
                         threshold: float) -> dict:
    """
    Analyze the feature plane for poisoning indicators.

    Checks:
      - Feature vector L2 norms (magnitude anomaly)
      - Per-dimension deviation from population mean
      - Score above detection threshold

    Returns dict: {node_id: {is_anomalous, score, norm, top_features, ...}}
    """
    x = data.x.numpy()
    num_nodes = data.num_nodes
    results = {}

    # Population statistics
    feat_mean = x.mean(axis=0)
    feat_std = x.std(axis=0) + 1e-8
    norms = np.linalg.norm(x, axis=1)
    norm_mean = norms.mean()
    norm_std = norms.std() + 1e-8

    for nid in range(num_nodes):
        score = scores[nid].item()
        node_features = x[nid]
        norm = norms[nid]

        # Z-scores for each feature dimension
        z_scores = (node_features - feat_mean) / feat_std
        top_dims = np.argsort(np.abs(z_scores))[::-1][:5]
        top_features = [
            {"name": FEATURE_NAMES[d], "value": round(float(node_features[d]), 4),
             "z_score": round(float(z_scores[d]), 2)}
            for d in top_dims if abs(z_scores[d]) > 1.5
        ]

        # Feature anomaly: score above threshold OR norm is outlier
        norm_z = (norm - norm_mean) / norm_std
        is_anomalous = score > threshold or norm_z > 2.0

        results[nid] = {
            "is_anomalous": bool(is_anomalous),
            "anomaly_score": round(score, 6),
            "feature_norm": round(float(norm), 4),
            "norm_z_score": round(float(norm_z), 2),
            "top_anomalous_features": top_features,
            "threshold": round(threshold, 6),
        }

    return results


# ═══════════════════════════════════════════════════════════════════════
# STRUCTURE PLANE
# ═══════════════════════════════════════════════════════════════════════

def verify_structure_plane(data: Data,
                           scores: torch.Tensor) -> dict:
    """
    Analyze the structure plane for topology poisoning indicators.

    Checks:
      - Degree distribution anomalies (unexpected connectivity)
      - Neighbor score correlation (poisoned nodes clustered together?)
      - Expected bipartite structure violations
      - Edge density around each node

    Returns dict: {node_id: {is_anomalous, degree, expected_degree, ...}}
    """
    edge_index = data.edge_index.numpy()
    num_nodes = data.num_nodes
    results = {}

    # Build adjacency
    adj = {i: set() for i in range(num_nodes)}
    for i in range(edge_index.shape[1]):
        s, d = int(edge_index[0, i]), int(edge_index[1, i])
        adj[s].add(d)
        adj[d].add(s)

    # Degree statistics
    degrees = np.array([len(adj[i]) for i in range(num_nodes)])
    deg_mean = degrees.mean()
    deg_std = degrees.std() + 1e-8

    # Expected degree in perfect bipartite graph
    # In spine-leaf: leaves connect to ALL spines, spines connect to ALL leaves
    # Detect roles by degree (spines have higher degree typically)
    median_deg = np.median(degrees)
    is_spine = degrees > median_deg  # rough heuristic

    n_spines = is_spine.sum()
    n_leaves = num_nodes - n_spines
    expected_spine_deg = max(1, n_leaves)
    expected_leaf_deg = max(1, n_spines)

    for nid in range(num_nodes):
        degree = int(degrees[nid])
        expected = expected_spine_deg if is_spine[nid] else expected_leaf_deg
        deg_z = (degree - deg_mean) / deg_std

        # Neighbor anomaly correlation
        neighbor_scores = [scores[n].item() for n in adj[nid]]
        if neighbor_scores:
            mean_nbr_score = np.mean(neighbor_scores)
            max_nbr_score = np.max(neighbor_scores)
            anomalous_neighbors = sum(1 for s in neighbor_scores if s > scores.mean().item() + 2 * scores.std().item())
        else:
            mean_nbr_score = 0.0
            max_nbr_score = 0.0
            anomalous_neighbors = 0

        # Structure anomaly criteria:
        # 1. Degree significantly different from expected
        # 2. High concentration of anomalous neighbors
        degree_anomaly = abs(degree - expected) > max(2, expected * 0.3)
        cluster_anomaly = anomalous_neighbors > len(adj[nid]) * 0.5 and anomalous_neighbors >= 2
        is_anomalous = degree_anomaly or cluster_anomaly

        results[nid] = {
            "is_anomalous": bool(is_anomalous),
            "degree": degree,
            "expected_degree": expected,
            "degree_z_score": round(float(deg_z), 2),
            "role": "spine" if is_spine[nid] else "leaf",
            "anomalous_neighbors": anomalous_neighbors,
            "total_neighbors": len(adj[nid]),
            "mean_neighbor_score": round(float(mean_nbr_score), 4),
            "max_neighbor_score": round(float(max_nbr_score), 4),
            "degree_anomaly": bool(degree_anomaly),
            "cluster_anomaly": bool(cluster_anomaly),
        }

    return results


# ═══════════════════════════════════════════════════════════════════════
# DUAL-PLANE CROSS-REFERENCE
# ═══════════════════════════════════════════════════════════════════════

def dual_plane_classify(feature_results: dict,
                        structure_results: dict) -> dict:
    """
    Cross-reference feature and structure plane results to produce
    a final poisoning classification per node.

    Returns dict: {node_id: {
        classification,
        feature_anomaly, structure_anomaly,
        confidence, severity
    }}
    """
    classifications = {}

    for nid in feature_results:
        feat = feature_results[nid]
        struct = structure_results.get(nid, {"is_anomalous": False})

        f_anom = feat["is_anomalous"]
        s_anom = struct["is_anomalous"]

        classification = CLASSIFICATIONS[(f_anom, s_anom)]

        # Confidence scoring
        if classification == "DUAL_POISONED":
            confidence = min(0.95, 0.7 + 0.1 * len(feat.get("top_anomalous_features", [])))
            severity = "CRITICAL"
        elif classification == "FEATURE_POISONED":
            confidence = min(0.90, 0.5 + 0.15 * feat["anomaly_score"] / max(feat["threshold"], 0.01))
            severity = "HIGH"
        elif classification == "STRUCTURE_POISONED":
            confidence = 0.65 + 0.1 * struct.get("anomalous_neighbors", 0)
            severity = "HIGH"
        else:
            confidence = 0.95
            severity = "NORMAL"

        classifications[nid] = {
            "classification": classification,
            "feature_anomaly": f_anom,
            "structure_anomaly": s_anom,
            "confidence": round(min(confidence, 0.99), 2),
            "severity": severity,
            "feature_score": feat["anomaly_score"],
            "feature_norm": feat["feature_norm"],
            "structure_degree": struct.get("degree", 0),
            "structure_role": struct.get("role", "unknown"),
            "top_features": feat.get("top_anomalous_features", []),
            "anomalous_neighbors": struct.get("anomalous_neighbors", 0),
        }

    return classifications


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

def verify(data: Data,
           scores: torch.Tensor,
           threshold: float,
           output_dir: str = "syntheticdata") -> dict:
    """
    Full dual-plane verification pipeline.

    1. Verify feature plane
    2. Verify structure plane
    3. Cross-reference for final classification
    4. Save verification report

    Returns: classifications dict
    """
    print("\n[dual-plane] ─── Dual-Plane Structural Poisoning Verification ───")

    # Feature plane
    print("[dual-plane] Verifying feature plane ...")
    feature_results = verify_feature_plane(data, scores, threshold)
    feat_flagged = sum(1 for v in feature_results.values() if v["is_anomalous"])
    print("[dual-plane]   Feature anomalies: %d / %d nodes" % (feat_flagged, data.num_nodes))

    # Structure plane
    print("[dual-plane] Verifying structure plane ...")
    structure_results = verify_structure_plane(data, scores)
    struct_flagged = sum(1 for v in structure_results.values() if v["is_anomalous"])
    print("[dual-plane]   Structure anomalies: %d / %d nodes" % (struct_flagged, data.num_nodes))

    # Cross-reference
    print("[dual-plane] Cross-referencing planes ...")
    classifications = dual_plane_classify(feature_results, structure_results)

    # Summary
    counts = {}
    for v in classifications.values():
        c = v["classification"]
        counts[c] = counts.get(c, 0) + 1

    print("[dual-plane] ─── Verification Results ───")
    for cls in ["CLEAN", "FEATURE_POISONED", "STRUCTURE_POISONED", "DUAL_POISONED"]:
        if cls in counts:
            print("[dual-plane]   %-22s : %d nodes" % (cls, counts[cls]))

    # Save report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "dual_plane_report.json")
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_nodes": data.num_nodes,
            "feature_anomalies": feat_flagged,
            "structure_anomalies": struct_flagged,
            "classifications": counts,
        },
        "nodes": {str(k): v for k, v in classifications.items()},
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("[dual-plane] Report saved: %s" % report_path)
    print("[dual-plane] ─── Verification Complete ───\n")

    return classifications
