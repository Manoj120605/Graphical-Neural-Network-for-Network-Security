import os
import sys
import json
import torch
import numpy as np
from langchain_core.tools import tool

# Ensure project root is importable
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Paths (match GNN pipeline output) ──
DATA_DIR = os.path.join(_PROJECT_ROOT, "syntheticdata")
GRAPH_PATH = os.path.join(DATA_DIR, "synthetic_graph.pt")
SCORES_PATH = os.path.join(DATA_DIR, "anomaly_scores.csv")
ALERTS_PATH = os.path.join(DATA_DIR, "alerts.json")

# ── Lazy caches ──
_data_cache = None
_scores_cache = None


def _load_data():
    global _data_cache
    if _data_cache is None and os.path.exists(GRAPH_PATH):
        _data_cache = torch.load(GRAPH_PATH, weights_only=False)
    return _data_cache


def _load_scores():
    global _scores_cache
    if _scores_cache is None and os.path.exists(SCORES_PATH):
        raw = np.loadtxt(SCORES_PATH, delimiter=",", skiprows=1)
        _scores_cache = {int(r[0]): {"score": r[1], "label": int(r[2])} for r in raw}
    return _scores_cache


def _load_alerts():
    if os.path.exists(ALERTS_PATH):
        with open(ALERTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("alerts", [])
    return []


# ── Feature names for explanation reports ──
FEATURE_NAMES = [
    "traffic_in", "traffic_out", "packet_loss", "latency",
    "crc_errors", "cpu_usage", "memory_usage", "connection_count",
    "interface_errors", "dropped_packets", "jitter", "link_utilization",
    "route_changes", "neighbor_count", "retransmissions", "queue_depth",
]


@tool
def query_anomalies(threshold_sigma: float = 2.0) -> str:
    """
    Query the GNN's latest anomaly detection results.
    Returns all nodes flagged above the given sigma threshold.
    Also includes active alerts from the alert pipeline.
    Args:
        threshold_sigma: Number of standard deviations above mean to flag (default: 2.0)
    """
    scores = _load_scores()
    if not scores:
        return ("No anomaly scores found. Run the pipeline first:\n"
                "  python run.py")

    values = np.array([v["score"] for v in scores.values()])
    mean, std = values.mean(), values.std()
    threshold = mean + threshold_sigma * std

    flagged = {
        nid: info for nid, info in scores.items()
        if info["score"] > threshold
    }

    if not flagged:
        return ("No nodes flagged above %.1f-sigma threshold (%.4f). "
                "Network appears clean." % (threshold_sigma, threshold))

    result = ("ALERT: %d node(s) flagged above %.1f-sigma "
              "(threshold=%.4f):\n\n" % (len(flagged), threshold_sigma, threshold))

    for nid, info in sorted(flagged.items(), key=lambda x: -x[1]["score"]):
        sigma_val = (info["score"] - mean) / (std + 1e-8)
        status = "ANOMALY" if info["label"] else "FALSE POSITIVE"
        result += ("  Node %3d: score=%.4f (%.1f-sigma) | ground_truth=%s\n"
                   % (nid, info["score"], sigma_val, status))

    # Append alert details
    alerts = _load_alerts()
    if alerts:
        result += "\nActive Alerts:\n"
        for a in alerts:
            result += ("  [%s] Node %d | score=%.4f | severity=%s | %s\n"
                       % (a["alert_type"], a["node_id"], a["anomaly_score"],
                          a["severity"], a["reason"]))
    
    return result


@tool
def explain_node(node_id: int) -> str:
    """
    Explain why a specific node was flagged as anomalous.
    Provides feature analysis, neighbor context, dual poisoning assessment,
    and security recommendations.
    Args:
        node_id: The integer ID of the node to explain
    """
    data = _load_data()
    scores = _load_scores()

    if data is None:
        return "Graph data not found. Run: python run.py"

    if node_id >= data.num_nodes:
        return "Node %d does not exist. Max node ID: %d" % (node_id, data.num_nodes - 1)

    features = data.x[node_id].numpy()
    label = data.y[node_id].item()

    # Neighbor info
    mask = data.edge_index[0] == node_id
    neighbors = data.edge_index[1][mask].tolist()

    # Compare to neighbor features (feature poisoning detection)
    if neighbors:
        neighbor_feats = data.x[neighbors].numpy()
        neighbor_mean = neighbor_feats.mean(axis=0)
        feature_deviation = float(np.linalg.norm(features - neighbor_mean))
    else:
        neighbor_mean = np.zeros_like(features)
        feature_deviation = 0.0

    score_info = scores.get(node_id, {}) if scores else {}
    score_val = score_info.get("score", 0)

    all_scores = np.array([v["score"] for v in scores.values()]) if scores else np.array([0])
    sigma = (score_val - all_scores.mean()) / (all_scores.std() + 1e-8)

    # Feature-level analysis
    feature_report = ""
    for i, fname in enumerate(FEATURE_NAMES[:data.x.shape[1]]):
        val = features[i]
        nb_val = neighbor_mean[i] if neighbors else 0.0
        diff = abs(val - nb_val)
        flag = " << ANOMALOUS" if diff > 3.0 else ""
        feature_report += "    %-20s: %8.4f  (neighbor_mean: %8.4f  diff: %6.4f)%s\n" % (
            fname, val, nb_val, diff, flag)

    # Dual poisoning assessment
    assessment = []
    poisoning_type = "UNKNOWN"

    if label == 1:
        assessment.append("CONFIRMED ANOMALY - node has injected/shifted feature distribution")

    # Feature poisoning indicators
    feature_norm = float(np.linalg.norm(features))
    neighbor_norms = [float(np.linalg.norm(data.x[nb].numpy())) for nb in neighbors] if neighbors else [0]
    mean_neighbor_norm = np.mean(neighbor_norms)
    if feature_norm > mean_neighbor_norm * 2.0:
        assessment.append("FEATURE POISONING SUSPECTED - feature norm (%.2f) is %.1fx neighbor mean (%.2f)" %
                          (feature_norm, feature_norm / (mean_neighbor_norm + 1e-8), mean_neighbor_norm))
        poisoning_type = "FEATURE"

    # Structure poisoning indicators
    degree = len(neighbors)
    all_degrees = [(data.edge_index[0] == i).sum().item() for i in range(data.num_nodes)]
    mean_degree = np.mean(all_degrees)
    std_degree = np.std(all_degrees)
    if degree > mean_degree + 2 * std_degree:
        assessment.append("STRUCTURE POISONING SUSPECTED - degree (%d) is %.1f-sigma above mean (%.1f)" %
                          (degree, (degree - mean_degree) / (std_degree + 1e-8), mean_degree))
        poisoning_type = "STRUCTURE" if poisoning_type == "UNKNOWN" else "DUAL"

    if sigma > 3:
        assessment.append("CRITICAL - anomaly score exceeds 3-sigma threshold")
    elif sigma > 2:
        assessment.append("HIGH - anomaly score exceeds 2-sigma threshold")

    if feature_deviation > 5:
        assessment.append("Strong feature deviation (%.2f L2) from neighborhood - possible feature injection" %
                          feature_deviation)

    if not assessment:
        assessment.append("Node appears within normal operational parameters")
        poisoning_type = "NONE"

    assessment_str = "\n  ".join(assessment)

    return """
Node Explanation Report
========================
Node ID          : %d
Anomaly Score    : %.4f (%.1f-sigma above mean)
Ground Truth     : %s
Poisoning Type   : %s
Degree           : %d neighbors
Feature Norm     : %.4f (neighbor mean: %.4f)
Feature Deviation: %.4f (L2 from neighborhood mean)

Feature Vector Analysis:
%s
Assessment:
  %s

Remediation Priority:
  1. Isolate node %d from production traffic
  2. Run config audit against last known-good baseline
  3. Check authentication logs for unauthorized access
  4. If feature poisoning: verify telemetry sensor integrity
  5. If structure poisoning: validate physical topology matches logical graph
""" % (node_id, score_val, sigma,
       "ANOMALY" if label else "NORMAL",
       poisoning_type, len(neighbors),
       feature_norm, mean_neighbor_norm,
       feature_deviation, feature_report,
       assessment_str, node_id)


@tool
def get_neighbors(node_id: int) -> str:
    """
    Get the 1-hop neighbor context of a node in the topology.
    Useful for structure poisoning analysis - checking if
    the node's connections look legitimate.
    Args:
        node_id: The integer ID of the node
    """
    data = _load_data()
    if data is None:
        return "Graph data not found. Run: python run.py"

    mask = data.edge_index[0] == node_id
    neighbors = data.edge_index[1][mask].tolist()
    scores = _load_scores()

    all_degrees = [(data.edge_index[0] == i).sum().item() for i in range(data.num_nodes)]
    mean_degree = np.mean(all_degrees)

    result = "Node %d has %d neighbors (network mean: %.1f):\n\n" % (
        node_id, len(neighbors), mean_degree)

    for nb in neighbors:
        score = scores[nb]["score"] if scores and nb in scores else 0
        label = scores[nb]["label"] if scores and nb in scores else 0
        status = "ANOMALY" if label == 1 else "Normal"
        if scores and nb in scores and score > np.mean([v["score"] for v in scores.values()]) + 2 * np.std([v["score"] for v in scores.values()]):
            status = "FLAGGED"
        result += "  Node %3d: score=%.4f | %s\n" % (nb, score, status)

    # Structure analysis
    if len(neighbors) > mean_degree + 2 * np.std(all_degrees):
        result += "\n  WARNING: Node degree is unusually high - possible structure poisoning"
    elif len(neighbors) < max(1, mean_degree - 2 * np.std(all_degrees)):
        result += "\n  WARNING: Node degree is unusually low - possible edge deletion attack"

    return result


@tool
def generate_remediation(node_id: int, anomaly_type: str = "config_drift") -> str:
    """
    Generate a remediation action plan for a flagged node.
    Supports dual poisoning defense strategies.
    Args:
        node_id: The node to remediate
        anomaly_type: One of 'feature_poisoning', 'structure_poisoning', 'dual_poisoning', 'config_drift', 'lateral_movement'
    """
    plans = {
        "feature_poisoning": """
REMEDIATION PLAN - Feature Poisoning Defense
===============================================
Target Node : %d
Attack Type : Feature Vector Manipulation
Defense     : Sensor Validation + Feature Sanitization

Steps:
  1. QUARANTINE telemetry input from node %d
  2. Cross-validate features with out-of-band SNMP polling
  3. Compare feature distribution against 30-day rolling baseline
  4. If confirmed tampered:
     a. Reset telemetry agent on the node
     b. Rotate credentials for monitoring interface
     c. Re-establish feature baseline from clean snapshot
  5. Re-run GNN inference with sanitized features
  6. Apply robust aggregation (median instead of mean) for this node

Estimated restore time : ~45 seconds
Operator confirmation  : REQUIRED
""" % (node_id, node_id),

        "structure_poisoning": """
REMEDIATION PLAN - Structure/Graph Poisoning Defense
======================================================
Target Node : %d
Attack Type : Topology Manipulation (edge injection/deletion)
Defense     : Physical Topology Verification + Edge Pruning

Steps:
  1. Activate out-of-band LLDP/CDP scan on all interfaces
  2. Compare physical neighbor table vs. logical graph adjacency
  3. For each unverified edge:
     a. Mark as SUSPICIOUS in graph
     b. Temporarily remove from GNN input
  4. Check for unauthorized new connections (rogue links)
  5. Verify spanning-tree topology matches expected design
  6. Re-run GNN inference with sanitized topology
  7. Rotate monitoring credentials on affected switches

Estimated containment : ~2 minutes
Operator confirmation  : CRITICAL - do not auto-approve
""" % node_id,

        "dual_poisoning": """
REMEDIATION PLAN - Dual Poisoning Defense (Feature + Structure)
=================================================================
Target Node : %d
Attack Type : Combined feature manipulation + topology tampering
Defense     : Full dual-plane verification

Phase 1 - Containment:
  1. Isolate node %d from production traffic
  2. Capture all traffic on uplink for forensic analysis

Phase 2 - Feature Defense:
  3. Cross-validate telemetry via secondary SNMP channel
  4. Compare feature norms against neighborhood baseline
  5. Apply robust feature aggregation (trimmed mean)

Phase 3 - Structure Defense:
  6. Run physical LLDP discovery scan
  7. Diff logical topology vs. physical connections
  8. Prune unverified edges from graph

Phase 4 - Recovery:
  9. Rebuild graph from verified physical topology
  10. Re-inject validated features only
  11. Re-run GNN with adversarial training enabled
  12. Monitor node for 24h before restoring to production

Estimated full recovery : ~5 minutes + 24h monitoring
Operator confirmation    : CRITICAL
""" % (node_id, node_id),

        "config_drift": """
REMEDIATION PLAN - Config Drift
=================================
Target Node : %d
Action Type : Configuration Rollback

Commands:
  1. conf t
       no acl CORP_TRUST permit 0.0.0.0/0
       acl CORP_TRUST permit 10.0.0.0/8
       ntp server 10.0.0.1
       logging host 10.0.0.1
     end
  2. Verify: show running-config | compare baseline

Estimated restore time : ~8 seconds
Operator confirmation  : REQUIRED
""" % node_id,

        "lateral_movement": """
REMEDIATION PLAN - Lateral Movement
======================================
Target Node : %d
Action Type : Isolation + Forensics

Commands:
  1. interface all -> shutdown (isolate node)
  2. capture traffic on uplink for 300s
  3. pull auth logs: show aaa sessions
  4. compare ACL hit counters vs. baseline

Estimated containment time : ~30 seconds
Operator confirmation       : REQUIRED
""" % node_id,
    }

    return plans.get(anomaly_type,
                     "Unknown anomaly type: %s. Supported: %s"
                     % (anomaly_type, ", ".join(plans.keys())))


@tool
def run_gnn_scan(refresh: bool = True) -> str:
    """
    Trigger a fresh GNN inference scan on the topology.
    Runs the full pipeline: generate -> detect -> visualize.
    Updates syntheticdata/ with fresh results.
    Args:
        refresh: Whether to regenerate graph and embeddings (True) or use cache (False)
    """
    global _data_cache, _scores_cache
    _data_cache = None
    _scores_cache = None

    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, os.path.join(_PROJECT_ROOT, "run.py")],
            capture_output=True, text=True, timeout=120,
            cwd=_PROJECT_ROOT,
        )
        if result.returncode == 0:
            return "GNN scan complete.\n\n%s" % result.stdout
        else:
            return "Scan failed:\nSTDOUT: %s\nSTDERR: %s" % (result.stdout, result.stderr)
    except Exception as e:
        return "Error running GNN scan: %s" % e
