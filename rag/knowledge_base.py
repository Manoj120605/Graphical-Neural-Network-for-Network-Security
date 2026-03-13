"""
knowledge_base.py
=================
Builds a ChromaDB vector store populated with:
  1. Per-node anomaly reports (from GNN pipeline outputs)
  2. Dual poisoning defense knowledge corpus
"""

import os
import sys
import json
import numpy as np
import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

CHROMA_DIR = os.path.join(_PROJECT_ROOT, "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Paths matching GNN pipeline output ──
DATA_DIR = os.path.join(_PROJECT_ROOT, "syntheticdata")
GRAPH_PATH = os.path.join(DATA_DIR, "synthetic_graph.pt")
SCORES_PATH = os.path.join(DATA_DIR, "anomaly_scores.csv")
ALERTS_PATH = os.path.join(DATA_DIR, "alerts.json")
ATTACK_LOG_PATH = os.path.join(DATA_DIR, "attack_log.json")
ATTACK_PATTERNS_DIR = os.path.join(_PROJECT_ROOT, "synthetic-attacks")


# ═══════════════════════════════════════════════════════════════════════
# DUAL POISONING DEFENSE KNOWLEDGE CORPUS
# ═══════════════════════════════════════════════════════════════════════

DUAL_POISONING_DOCS = [
    # ── 1. Overview ──
    Document(page_content="""
Dual Poisoning Defense Overview

Dual poisoning is a sophisticated adversarial attack against Graph Neural Networks
where an attacker simultaneously manipulates BOTH the node feature vectors AND
the graph topology (edges). This makes detection significantly harder than either
attack alone because:

1. Feature Poisoning alone shifts a node's feature vector away from normal, but
   the GNN can still leverage correct neighborhood structure to detect it.
2. Structure Poisoning alone adds or removes edges to confuse neighborhood
   aggregation, but clean features help the GNN remain resilient.
3. Dual Poisoning combines both, creating a coordinated attack that undermines
   both the feature channel and the structural channel simultaneously.

In the AutoNet-GNN context, this means an attacker could:
  - Inject false telemetry (CPU, traffic, latency) into a compromised node
  - AND reroute network connections so the node appears connected to legitimate
    devices, masking the anomaly from the GNN's neighborhood aggregation.

The GraphSAGE model aggregates neighbor features via mean pooling, so if both the
node's own features AND its neighbor connections are manipulated, the embedding
will appear normal even though the node is compromised.
""", metadata={"type": "knowledge", "topic": "dual_poisoning_overview"}),

    # ── 2. Feature Poisoning ──
    Document(page_content="""
Feature Poisoning Attack on GNNs

Feature poisoning targets the node attribute matrix X. The attacker modifies
feature values of one or more nodes to either:
  A) Evade detection: make a malicious node look normal
  B) Cause false positives: make clean nodes look anomalous

In network security, feature poisoning examples include:
  - Spoofing telemetry: reporting artificially low CPU/memory usage
  - Masking packet loss: suppressing error counters on interfaces
  - Inflating traffic metrics: making a node appear busy when it is idle
  - Zeroing CRC errors: hiding hardware degradation

Detection methods for feature poisoning:
  1. Feature Norm Checking: compute L2 norm of each node's feature vector
     and compare to neighborhood mean. Poisoned nodes often have norms that
     deviate significantly from their neighbors.
  2. Feature Distribution Analysis: check if individual feature values fall
     outside the expected distribution (z-score > 3 for each dimension).
  3. Temporal Consistency: compare current features against historical
     baseline. Sudden large shifts suggest injection.
  4. Cross-Validation: poll the same metrics via a secondary channel (e.g.,
     SNMP vs. streaming telemetry) and compare values.

Key indicator: feature_norm(node) / mean(feature_norm(neighbors)) > 2.0
""", metadata={"type": "knowledge", "topic": "feature_poisoning"}),

    # ── 3. Structure/Graph Poisoning ──
    Document(page_content="""
Structure (Graph) Poisoning Attack on GNNs

Structure poisoning targets the adjacency matrix (edge_index). The attacker
manipulates the graph topology by:
  A) Edge injection: adding fake connections to normal nodes so a malicious
     node's neighborhood looks clean after aggregation
  B) Edge deletion: removing connections to isolate a node or reduce the
     GNN's ability to propagate anomaly signals
  C) Edge rewiring: redirecting connections to change community structure

In network security, structure poisoning examples include:
  - Rogue link injection: physically or logically connecting unauthorized devices
  - ARP/MAC spoofing: making a switch think a rogue device is a legitimate neighbor
  - LLDP/CDP spoofing: faking layer-2 neighbor discovery messages
  - BGP hijacking: advertising false routes to alter logical topology
  - Spanning-tree manipulation: changing the active topology

Detection methods for structure poisoning:
  1. Degree Analysis: compare node degree against network mean +/- 2*std.
     Sudden degree changes indicate edge injection or deletion.
  2. Physical vs. Logical Verification: compare LLDP neighbor tables against
     the logical graph. Unverified edges are suspicious.
  3. Spectral Analysis: compute graph Laplacian eigenvalues. Large shifts in
     the spectral gap indicate structural perturbation.
  4. Edge Weight Validation: in weighted graphs, check if new edges have
     anomalous weights compared to the distribution.

Key indicator: degree(node) > mean_degree + 2 * std_degree
""", metadata={"type": "knowledge", "topic": "structure_poisoning"}),

    # ── 4. Dual Poisoning Detection Strategy ──
    Document(page_content="""
Dual Poisoning Detection Strategy

When both feature and structure poisoning occur simultaneously, single-channel
detection methods are insufficient. AutoNet-GNN uses a multi-signal approach:

Signal 1 - Embedding Deviation Score (primary):
  score = L2(node_embedding, mean(neighbor_embeddings))
  Uses the 2-sigma rule: threshold = mean(scores) + 2 * std(scores)
  This catches most anomalies but can be evaded by dual poisoning.

Signal 2 - Feature Norm Ratio:
  ratio = L2_norm(node_features) / mean(L2_norm(neighbor_features))
  If ratio > 2.0, feature poisoning is suspected.
  If the embedding score is LOW but feature norm ratio is HIGH, the structure
  has likely been manipulated to mask the feature anomaly.

Signal 3 - Degree Deviation:
  z_degree = (degree(node) - mean_degree) / std_degree
  If z_degree > 2.0, structure poisoning is suspected.
  Combined with a high feature norm ratio, this indicates dual poisoning.

Signal 4 - Cross-Validation Score:
  Re-run GNN inference with the node's edges removed (isolation test).
  If the node's embedding changes dramatically, the original edges were
  masking an anomaly (structure poisoning confirmed).

Decision Matrix:
  | Embedding Score | Feature Norm | Degree | Verdict              |
  |:----------------|:-------------|:-------|:---------------------|
  | HIGH            | HIGH         | Normal | Feature poisoning    |
  | HIGH            | Normal       | HIGH   | Structure poisoning  |
  | HIGH            | HIGH         | HIGH   | Dual poisoning       |
  | LOW             | HIGH         | HIGH   | Masked dual attack   |
  | LOW             | Normal       | Normal | Clean node           |
""", metadata={"type": "knowledge", "topic": "dual_detection_strategy"}),

    # ── 5. Defense Mechanisms ──
    Document(page_content="""
Defense Mechanisms Against Dual Poisoning

1. Robust Aggregation (Feature Defense):
   Replace mean aggregation in GraphSAGE with trimmed mean or median.
   This reduces the impact of poisoned neighbor features on the target
   node's embedding. The trimmed mean discards the top/bottom 10%% of
   neighbor feature values before averaging.

2. Graph Sanitization (Structure Defense):
   Before GNN inference, validate the graph structure:
   a) Remove edges not confirmed by physical LLDP/CDP discovery
   b) Prune edges where both endpoints have high anomaly scores
   c) Apply spectral filtering: remove edges that cause the largest
      change in the graph Laplacian's Fiedler value

3. Adversarial Training:
   Train the GraphSAGE model with adversarial examples:
   a) During training, randomly perturb 5-10%% of feature vectors
   b) Randomly add/remove 5%% of edges
   c) Train the model to still correctly identify anomalies despite
      these perturbations
   This produces a model robust to both poisoning types.

4. Dual-Plane Verification:
   Maintain two independent graph representations:
   a) Control Plane: topology from routing protocols (OSPF, BGP)
   b) Data Plane: topology from actual traffic flows (NetFlow, sFlow)
   Discrepancies between planes indicate structure poisoning.

5. Feature Provenance Tracking:
   Record the source and timestamp of each feature dimension.
   If a feature is sourced only from the node itself (self-reported),
   cross-validate with passive measurements before trusting it.
""", metadata={"type": "knowledge", "topic": "defense_mechanisms"}),

    # ── 6. GraphSAGE Vulnerability Analysis ──
    Document(page_content="""
Why GraphSAGE is Vulnerable to Dual Poisoning

GraphSAGE uses the following aggregation in each layer:
  h_v = sigma(W * CONCAT(h_v, AGG({h_u : u in N(v)})))

Where AGG is typically MEAN. This creates two attack surfaces:

Attack Surface 1 - Feature Channel (h_v):
  The node's own feature vector h_v is directly concatenated into the
  embedding. An attacker controlling the node can set h_v to any value.
  With mean aggregation, a single poisoned feature vector shifts the
  entire neighborhood's embedding.

Attack Surface 2 - Structure Channel (N(v)):
  The neighbor set N(v) determines WHICH nodes participate in aggregation.
  By adding edges to clean nodes, an attacker dilutes its own anomalous
  signal with normal features. By removing edges to anomalous neighbors,
  it prevents anomaly signal propagation.

Combined Attack (Dual Poisoning):
  The attacker simultaneously:
  1. Sets its own features close to the neighborhood mean (stealth)
  2. Adds edges to high-degree clean hub nodes (dilution)
  3. Removes edges to other suspicious nodes (isolation)

  This produces a node embedding that passes the L2 deviation test
  because both its features AND its neighborhood look clean.

  The key insight is that the L2 deviation score measures the distance
  between a node's embedding and its CURRENT neighbors' mean embedding.
  If the neighbors are manipulated to match, the score drops below
  threshold even though the node is compromised.
""", metadata={"type": "knowledge", "topic": "graphsage_vulnerability"}),

    # ── 7. Remediation Playbooks ──
    Document(page_content="""
Remediation Playbooks for Dual Poisoning

Playbook A - Feature Poisoning Confirmed:
  Priority: HIGH | Estimated time: 45 seconds
  1. Quarantine telemetry input from the affected node
  2. Cross-validate features via out-of-band SNMP polling
  3. Compare feature distribution against 30-day rolling baseline
  4. Reset telemetry agent on the compromised node
  5. Rotate monitoring interface credentials
  6. Re-establish feature baseline from verified clean snapshot
  7. Re-run GNN inference with sanitized features
  8. Apply median aggregation as temporary safeguard

Playbook B - Structure Poisoning Confirmed:
  Priority: CRITICAL | Estimated time: 2 minutes
  1. Activate out-of-band LLDP/CDP discovery scan
  2. Compare physical neighbor table vs logical graph edges
  3. Mark all unverified edges as SUSPICIOUS
  4. Temporarily remove suspicious edges from GNN input
  5. Check for unauthorized physical connections (rogue switches)
  6. Verify spanning-tree topology against expected design
  7. Re-run GNN inference with sanitized topology
  8. Rotate monitoring credentials on all affected switches

Playbook C - Dual Poisoning Confirmed:
  Priority: CRITICAL | Estimated time: 5 minutes + 24h monitoring
  1. IMMEDIATE: Isolate node from production traffic
  2. Execute Playbook A steps 1-5 (feature defense)
  3. Execute Playbook B steps 1-6 (structure defense)
  4. Rebuild graph from verified physical topology only
  5. Re-inject only validated features (cross-checked via SNMP)
  6. Enable adversarial training mode for next inference cycle
  7. Set node to PROBATION state: monitor for 24 hours
  8. Require dual-operator approval before restoring to production
""", metadata={"type": "knowledge", "topic": "remediation_playbooks"}),

    # ── 8. Real-World Attack Scenarios ──
    Document(page_content="""
Real-World Dual Poisoning Attack Scenarios in Network Security

Scenario 1 - Insider Threat with Compromised Switch:
  An insider gains access to a network switch and:
  - Modifies SNMP MIBs to report false interface counters (feature poisoning)
  - Configures trunk links to unauthorized VLANs (structure poisoning)
  Detection: The GNN should flag abnormal L2 neighbor count combined with
  suspiciously clean interface error metrics.

Scenario 2 - Supply Chain Attack on Network Monitoring:
  Malware in a monitoring agent modifies telemetry before collection:
  - Zeroes out CRC errors and packet loss (feature poisoning)
  - Injects fake LLDP frames to appear connected to core switches (structure)
  Detection: Cross-validate LLDP neighbor tables with physical port-channel
  membership. Feature provenance tracking catches self-reported-only metrics.

Scenario 3 - Advanced Persistent Threat (APT) Lateral Movement:
  An APT actor compromises multiple nodes and:
  - Makes each compromised node report features similar to its neighbors (stealth)
  - Creates logical tunnels between compromised nodes (structure manipulation)
  Detection: Cluster analysis on embeddings reveals a tight cluster of nodes
  with suspiciously similar features that shouldn't be similar given their
  physical separation in the topology.

Scenario 4 - Ransomware Pre-Positioning:
  Before activating ransomware, the attacker:
  - Gradually reduces reported traffic metrics to avoid sudden-change detection
  - Inserts routing table entries to bypass firewall inspection
  Detection: Temporal analysis detects the slow feature drift. Dual-plane
  verification (control vs data plane) catches the routing manipulation.
""", metadata={"type": "knowledge", "topic": "attack_scenarios"}),
]


# ═══════════════════════════════════════════════════════════════════════
# BUILD / LOAD
# ═══════════════════════════════════════════════════════════════════════

def build_knowledge_base(data_path=None, scores_path=None) -> Chroma:
    """
    Build ChromaDB vector store from GNN outputs + dual poisoning knowledge.
    """
    if data_path is None:
        data_path = GRAPH_PATH
    if scores_path is None:
        scores_path = SCORES_PATH

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    documents = []

    # ── 1. Per-node anomaly reports from GNN ──
    if os.path.exists(data_path):
        data = torch.load(data_path, weights_only=False)

        scores = None
        if os.path.exists(scores_path):
            raw = np.loadtxt(scores_path, delimiter=",", skiprows=1)
            scores = {int(r[0]): (r[1], int(r[2])) for r in raw}

        mean_score = np.mean([s[0] for s in scores.values()]) if scores else 0
        std_score = np.std([s[0] for s in scores.values()]) if scores else 1

        for node_id in range(data.num_nodes):
            features = data.x[node_id].numpy().tolist()
            label = data.y[node_id].item()

            score_val = scores[node_id][0] if scores and node_id in scores else 0
            sigma = (score_val - mean_score) / (std_score + 1e-8)

            mask = data.edge_index[0] == node_id
            neighbors = data.edge_index[1][mask].tolist()
            feature_norm = float(np.linalg.norm(features))

            content = """
Node ID: %d
Label: %s
Anomaly Score: %.4f (%.2f-sigma)
Degree (connections): %d
Neighbors: %s
Feature norm: %.4f
Feature mean: %.4f
Feature std: %.4f
Feature max: %.4f
Feature min: %.4f
Status: %s
""" % (node_id,
       "ANOMALY" if label == 1 else "NORMAL",
       score_val, sigma,
       len(neighbors),
       str(neighbors[:10]),
       feature_norm,
       np.mean(features), np.std(features),
       max(features), min(features),
       "CRITICAL - confirmed anomaly, possible dual poisoning target" if label == 1
       else "Normal operation")

            documents.append(Document(
                page_content=content,
                metadata={
                    "node_id": node_id, "label": label,
                    "score": score_val, "sigma": sigma,
                    "type": "node_report",
                }
            ))

    # ── 2. Alert reports ──
    if os.path.exists(ALERTS_PATH):
        with open(ALERTS_PATH, "r", encoding="utf-8") as f:
            alerts = json.load(f).get("alerts", [])
        for alert in alerts:
            content = """
Alert Report
Alert Type: %s
Node ID: %d
Anomaly Score: %.4f
Threshold: %.4f
Severity: %s
Reason: %s
Timestamp: %s
""" % (alert["alert_type"], alert["node_id"], alert["anomaly_score"],
       alert["threshold"], alert["severity"], alert["reason"],
       alert.get("timestamp", "unknown"))
            documents.append(Document(
                page_content=content,
                metadata={"type": "alert", "node_id": alert["node_id"]},
            ))

    # ── 3. Dual poisoning defense knowledge corpus ──
    documents.extend(DUAL_POISONING_DOCS)

    # ── 4. Synthetic attack pattern knowledge ──
    if os.path.isdir(ATTACK_PATTERNS_DIR):
        atk_count = 0
        for fname in sorted(os.listdir(ATTACK_PATTERNS_DIR)):
            if fname.startswith("ATK-") and fname.endswith(".json"):
                fpath = os.path.join(ATTACK_PATTERNS_DIR, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    atk = json.load(f)

                iocs = "\n".join("  - %s" % i for i in atk.get("indicators_of_compromise", []))
                det_rules = json.dumps(atk.get("detection_rules", {}), indent=2)
                sim_info = json.dumps(atk.get("simulation", {}), indent=2)

                content = """
Attack Pattern: %s
ID: %s
Category: %s
MITRE ATT&CK: %s
Severity: %s
Description: %s

Indicators of Compromise:
%s

Detection Rules:
%s

Simulation Parameters:
%s
""" % (atk.get("name", "Unknown"), atk.get("id", "???"),
       atk.get("category", "Unknown"), atk.get("mitre_att_ck", ""),
       atk.get("severity", "unknown"), atk.get("description", ""),
       iocs, det_rules, sim_info)

                documents.append(Document(
                    page_content=content,
                    metadata={
                        "type": "attack_pattern",
                        "attack_id": atk.get("id", ""),
                        "category": atk.get("category", ""),
                        "severity": atk.get("severity", ""),
                        "mitre": atk.get("mitre_att_ck", ""),
                    },
                ))
                atk_count += 1
        print("[kb] Indexed %d attack patterns" % atk_count)

    # ── 5. Attack simulation log ──
    if os.path.exists(ATTACK_LOG_PATH):
        with open(ATTACK_LOG_PATH, "r", encoding="utf-8") as f:
            atk_log = json.load(f)
        for event in atk_log.get("attacks", []):
            victims = ", ".join(event.get("victim_nodes", []))
            iocs = "\n".join("  - %s" % i for i in event.get("indicators", []))
            content = """
Simulated Attack Event
Attack: %s (%s)
Category: %s
Severity: %s
MITRE ATT&CK: %s
Victim Nodes: %s
Timestamp: %s

Expected Indicators:
%s
""" % (event.get("attack_name", "Unknown"), event.get("attack_id", "???"),
       event.get("category", ""), event.get("severity", ""),
       event.get("mitre_att_ck", ""), victims,
       event.get("timestamp", "unknown"), iocs)

            documents.append(Document(
                page_content=content,
                metadata={
                    "type": "attack_simulation",
                    "attack_id": event.get("attack_id", ""),
                },
            ))
        print("[kb] Indexed %d attack simulation events" % len(atk_log.get("attacks", [])))

    # ── Build ──
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    print("[kb] Knowledge base built: %d total documents indexed" % len(documents))
    return vectorstore


def load_knowledge_base() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
