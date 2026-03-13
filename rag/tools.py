import os
import sys
import json
import torch
import numpy as np
from langchain_core.tools import tool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GNN'))

# ── Lazy load GNN modules ──
_data_cache = None
_scores_cache = None

def _load_data():
    global _data_cache
    if _data_cache is None and os.path.exists("synthetic_graph.pt"):
        _data_cache = torch.load("synthetic_graph.pt", weights_only=False)
    return _data_cache

def _load_scores():
    global _scores_cache
    if _scores_cache is None and os.path.exists("anomaly_scores.csv"):
        raw = np.loadtxt("anomaly_scores.csv", delimiter=",", skiprows=1)
        _scores_cache = {int(r[0]): {"score": r[1], "label": int(r[2])} for r in raw}
    return _scores_cache


@tool
def query_anomalies(threshold_sigma: float = 2.0) -> str:
    """
    Query the GNN's latest anomaly detection results.
    Returns all nodes flagged above the given sigma threshold.
    Args:
        threshold_sigma: Number of standard deviations above mean to flag (default: 2.0)
    """
    scores = _load_scores()
    if not scores:
        return "No anomaly scores found. Run detect_anomalies.py first."
    
    values = np.array([v["score"] for v in scores.values()])
    mean, std = values.mean(), values.std()
    threshold  = mean + threshold_sigma * std
    
    flagged = {
        nid: info for nid, info in scores.items()
        if info["score"] > threshold
    }
    
    if not flagged:
        return f"No nodes flagged above {threshold_sigma}σ threshold ({threshold:.4f}). Network appears clean."
    
    result = f"🚨 {len(flagged)} node(s) flagged above {threshold_sigma}σ (threshold={threshold:.4f}):\n\n"
    for nid, info in sorted(flagged.items(), key=lambda x: -x[1]["score"]):
        sigma_val = (info["score"] - mean) / (std + 1e-8)
        result += f"  Node {nid:>3}: score={info['score']:.4f} ({sigma_val:.1f}σ) | label={'ANOMALY' if info['label'] else 'NORMAL'}\n"
    
    return result


@tool
def explain_node(node_id: int) -> str:
    """
    Explain why a specific node was flagged as anomalous.
    Provides feature analysis, neighbor context, and security assessment.
    Args:
        node_id: The integer ID of the node to explain
    """
    data   = _load_data()
    scores = _load_scores()
    
    if data is None:
        return "Graph data not found. Run generate_topology.py first."
    
    if node_id >= data.num_nodes:
        return f"Node {node_id} does not exist. Max node ID: {data.num_nodes - 1}"
    
    features  = data.x[node_id].numpy()
    label     = data.y[node_id].item()
    
    # Neighbor info
    mask      = data.edge_index[0] == node_id
    neighbors = data.edge_index[1][mask].tolist()
    
    # Compare to neighbor features
    if neighbors:
        neighbor_feats = data.x[neighbors].numpy()
        neighbor_mean  = neighbor_feats.mean(axis=0)
        deviation      = np.linalg.norm(features - neighbor_mean)
    else:
        deviation = 0.0
    
    score_info = scores.get(node_id, {}) if scores else {}
    score_val  = score_info.get("score", 0)
    
    all_scores = np.array([v["score"] for v in scores.values()]) if scores else np.array([0])
    sigma = (score_val - all_scores.mean()) / (all_scores.std() + 1e-8)
    
    assessment = []
    if label == 1:
        assessment.append("⚠ CONFIRMED ANOMALY — node was injected with shifted feature distribution")
    if sigma > 3:
        assessment.append("🔴 CRITICAL — anomaly score exceeds 3σ threshold")
    elif sigma > 2:
        assessment.append("🟠 HIGH — anomaly score exceeds 2σ threshold")
    if deviation > 5:
        assessment.append(f"Feature vector deviates {deviation:.2f} from neighbor mean — strong structural mismatch")
    if len(neighbors) > 10:
        assessment.append(f"High-degree hub node ({len(neighbors)} connections) — impact radius is significant")
    
    if not assessment:
        assessment.append("✅ Node appears within normal operational parameters")
    
    return f"""
Node Explanation Report
========================
Node ID       : {node_id}
Anomaly Score : {score_val:.4f} ({sigma:.1f}σ above mean)
Ground Truth  : {"ANOMALY" if label else "NORMAL"}
Degree        : {len(neighbors)} neighbors → {neighbors[:8]}{"..." if len(neighbors) > 8 else ""}

Feature Statistics:
  Mean  : {features.mean():.4f}
  Std   : {features.std():.4f}
  Max   : {features.max():.4f}
  Min   : {features.min():.4f}

Neighbor Deviation (L2): {deviation:.4f}

Assessment:
{"  " + chr(10) + "  ".join(assessment)}

Recommendation:
  1. Isolate node {node_id} from production traffic
  2. Run config audit against last known-good baseline
  3. Check authentication logs for unauthorized access
  4. Compare physical interface state vs. logical topology
"""


@tool
def get_neighbors(node_id: int) -> str:
    """
    Get the 1-hop neighbor context of a node in the topology.
    Args:
        node_id: The integer ID of the node
    """
    data = _load_data()
    if data is None:
        return "Graph data not found."
    
    mask      = data.edge_index[0] == node_id
    neighbors = data.edge_index[1][mask].tolist()
    scores    = _load_scores()
    
    result = f"Node {node_id} has {len(neighbors)} neighbors:\n\n"
    for nb in neighbors:
        score = scores[nb]["score"] if scores and nb in scores else "N/A"
        label = scores[nb]["label"] if scores and nb in scores else "?"
        status = "⚠ ANOMALY" if label == 1 else "✓ Normal"
        result += f"  → Node {nb:>3}: score={score:.4f if isinstance(score, float) else score} | {status}\n"
    
    return result


@tool
def generate_remediation(node_id: int, anomaly_type: str = "config_drift") -> str:
    """
    Generate a remediation action plan for a flagged node.
    Args:
        node_id: The node to remediate
        anomaly_type: One of 'config_drift', 'lateral_movement', 'hardware_degradation', 'graph_poisoning'
    """
    remediation_plans = {
        "config_drift": f"""
REMEDIATION PLAN — Config Drift
=================================
Target Node : {node_id}
Action Type : Configuration Rollback

Commands:
  1. conf t
       no acl CORP_TRUST permit 0.0.0.0/0
       acl CORP_TRUST permit 10.0.0.0/8
       ntp server 10.0.0.1
       snmp-server community public ro
       logging host 10.0.0.1
     end

  2. Verify: show running-config | compare baseline_rev_47

Estimated restore time : ~8 seconds
Operator confirmation  : REQUIRED

  [ APPROVE ]  [ REJECT ]  [ ESCALATE ]
""",
        "lateral_movement": f"""
REMEDIATION PLAN — Lateral Movement
======================================
Target Node : {node_id}
Action Type : Isolation + Forensics

Commands:
  1. interface all → shutdown (isolate node)
  2. capture traffic on uplink for 300s
  3. pull auth logs: show aaa sessions node {node_id}
  4. compare ACL hit counters vs. baseline

Estimated containment time : ~30 seconds
Operator confirmation       : REQUIRED
""",
        "hardware_degradation": f"""
REMEDIATION PLAN — Hardware Degradation
=========================================
Target Node : {node_id}
Action Type : Maintenance Scheduling

Commands:
  1. show interfaces | grep CRC
  2. show environment power
  3. show environment temperature
  4. schedule maintenance-window node {node_id} in 2h

Urgency : MEDIUM (monitor for 1h before action)
""",
        "graph_poisoning": f"""
REMEDIATION PLAN — Graph Poisoning
=====================================
Target Node : {node_id}
Action Type : Dual-Plane Verification + Pruning

Commands:
  1. Activate out-of-band LLDP scan
  2. Compare physical neighbors vs. logical graph
  3. Prune edges with no physical confirmation
  4. Rotate monitoring credentials
  5. Re-seed trust anchor from hardware attestation

Operator confirmation : CRITICAL — do not auto-approve
"""
    }
    
    return remediation_plans.get(anomaly_type, f"Unknown anomaly type: {anomaly_type}")


@tool
def run_gnn_scan(refresh: bool = True) -> str:
    """
    Trigger a fresh GNN inference scan on the topology.
    Re-runs detect_anomalies.py and updates anomaly_scores.csv.
    Args:
        refresh: Whether to regenerate embeddings (True) or use cache (False)
    """
    global _data_cache, _scores_cache
    
    try:
        import subprocess
        result = subprocess.run(
            ["python", "GNN/detect_anomalies.py"],
            capture_output=True, text=True, timeout=60
        )
        _scores_cache = None  # bust cache
        
        if result.returncode == 0:
            return f"✔ GNN scan complete.\n\n{result.stdout}"
        else:
            return f"❌ Scan failed:\n{result.stderr}"
    except Exception as e:
        return f"Error running GNN scan: {e}"
