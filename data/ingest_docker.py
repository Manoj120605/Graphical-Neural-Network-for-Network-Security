"""
ingest_docker.py
================
Bridges live Docker-based network simulation with the GraphSAGE
anomaly-detection pipeline.

Pipeline
    1. Ingest   - discover containers & poll real 16D telemetry
    2. Graph    - build strict bipartite spine-leaf topology
    3. Normalise- z-score normalise features across nodes
    4. Convert  - NetworkX -> PyTorch Geometric Data object
    5. Export   - save to syntheticdata/synthetic_graph.pt

Supports two telemetry modes:
    A) Live     : 16D feature vectors from get_telemetry()
    B) Attack   : attack_telemetry override from attack_simulator
"""

import os
import sys
import time

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# ---------------------------------------------------------------------------
# Allow importing from the project root so Node_Creation is reachable
# regardless of the working directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# NOTE: Node_Creation.autonet_core imports the Docker SDK at module level.
# We defer that import to ingest() so the pure helper functions
# (_build_bipartite_graph, _build_features, _to_pyg_data) can be imported
# and tested without Docker installed.

# ---- Constants ------------------------------------------------------------
FEATURE_DIM = 16          # must match GraphSAGEEncoder(in_dim=16)
OUTPUT_DIR = "syntheticdata"
OUTPUT_FILE = "synthetic_graph.pt"


# ---------------------------------------------------------------------------
# Step 2 helper -- bipartite spine-leaf graph
# ---------------------------------------------------------------------------
def _build_bipartite_graph(node_names: list[str]) -> nx.Graph:
    """
    Build a strict bipartite graph from container names.

    Rules
        A) Every leaf connects to every spine.
        B) No leaf-leaf or spine-spine edges.

    Parameters
    ----------
    node_names : list[str]
        Container names containing either ``"spine"`` or ``"leaf"``.

    Returns
    -------
    G : nx.Graph
        Undirected bipartite graph with integer node IDs and a ``name``
        attribute on each node.
    """
    spines = [n for n in node_names if "spine" in n.lower()]
    leaves = [n for n in node_names if "leaf" in n.lower()]

    G = nx.Graph()

    # Add all nodes with their original container name stored as attribute
    for idx, name in enumerate(node_names):
        role = "spine" if "spine" in name.lower() else "leaf"
        G.add_node(idx, name=name, role=role)

    # Build a name -> index lookup
    name_to_idx = {name: idx for idx, name in enumerate(node_names)}

    # Full bipartite edges: every leaf <-> every spine
    for leaf_name in leaves:
        for spine_name in spines:
            G.add_edge(name_to_idx[leaf_name], name_to_idx[spine_name])

    return G


# ---------------------------------------------------------------------------
# Step 3 helper -- build feature matrix from 16D telemetry + normalise
# ---------------------------------------------------------------------------
def _build_features(telemetry, node_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the feature matrix from real 16D telemetry vectors returned by
    get_telemetry().  Apply z-score normalisation per-feature so the GNN
    receives normalised inputs with variance across nodes.

    Accepts telemetry in two formats:
      A) List of [node_name, [16D feature list]]  (from get_telemetry)
      B) List of dicts with 'name' and 'features' keys (from attack_simulator)

    Labels are set to 0 for all nodes (live mode has no ground truth).

    Parameters
    ----------
    telemetry : list
        Telemetry data in format A or B.
    node_names : list[str]
        Ordered master list of container names (defines row ordering).

    Returns
    -------
    features : np.ndarray, shape (N, 16)
    labels   : np.ndarray, shape (N,)  -- all zeros (no ground truth in live mode)
    """
    # Map name -> feature vector for O(1) lookup
    feat_map: dict[str, list[float]] = {}
    for entry in telemetry:
        if isinstance(entry, dict):
            # Format B: dict with 'name' and 'features' keys
            name = entry['name']
            feats = entry.get('features', [0.0] * FEATURE_DIM)
        else:
            # Format A: [name, features_or_scalar]
            name = entry[0]
            feats = entry[1]

        # Handle legacy single-bit format (scalar) and new 16D format (list)
        if isinstance(feats, (int, float)):
            vec = [0.0] * FEATURE_DIM
            vec[0] = float(feats)
            feat_map[name] = vec
        else:
            feat_map[name] = [float(f) for f in feats]

    num_nodes = len(node_names)
    features = np.zeros((num_nodes, FEATURE_DIM), dtype=np.float32)

    for idx, name in enumerate(node_names):
        vec = feat_map.get(name, [0.0] * FEATURE_DIM)
        features[idx, :len(vec)] = vec[:FEATURE_DIM]

    # Z-score normalise per feature (so each dimension has mean~0, std~1)
    # This prevents large-magnitude features (rx_bytes) from dominating
    for col in range(FEATURE_DIM):
        col_data = features[:, col]
        mu = col_data.mean()
        sigma = col_data.std()
        if sigma > 1e-8:
            features[:, col] = (col_data - mu) / sigma
        else:
            # All same value -> zero it out (no discriminative signal)
            features[:, col] = 0.0

    # Live mode: no ground-truth labels (we do NOT know which nodes are anomalous)
    labels = np.zeros(num_nodes, dtype=np.int64)

    return features, labels


# ---------------------------------------------------------------------------
# Step 4 helper -- PyG conversion
# ---------------------------------------------------------------------------
def _to_pyg_data(G: nx.Graph, features: np.ndarray,
                 labels: np.ndarray) -> Data:
    """
    Convert the NetworkX bipartite graph and NumPy arrays into a PyTorch
    Geometric ``Data`` object, matching the conventions used by
    ``generate_topology.py`` and consumed by ``detect_anomalies.py``.
    """
    pyg = from_networkx(G)
    pyg.x = torch.tensor(features, dtype=torch.float32)
    pyg.y = torch.tensor(labels, dtype=torch.long)
    pyg.train_mask = torch.ones(pyg.num_nodes, dtype=torch.bool)
    pyg.anomaly_mask = pyg.y.bool()
    # Mark this as live data (no injected ground truth)
    pyg.is_live = True
    return pyg


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def ingest(output_dir: str = OUTPUT_DIR,
           attack_telemetry: list[dict] | None = None,
           attack_log: list[dict] | None = None) -> Data:
    """
    End-to-end live-ingestion pipeline.

    1. Discover Docker containers (spine/leaf).
    2. Poll real 16D telemetry from each container.
    3. Build bipartite spine-leaf graph.
    4. Z-score normalise features.
    5. Convert to PyG ``Data`` and save to disk.

    Parameters
    ----------
    output_dir : str
        Directory to save the PyG graph.
    attack_telemetry : list[dict], optional
        Override telemetry from the attack simulator. If provided, this
        is merged with (or replaces) the live telemetry.
    attack_log : list[dict], optional
        Attack log to save alongside the graph.

    Returns
    -------
    data : torch_geometric.data.Data
    """
    os.makedirs(output_dir, exist_ok=True)
    start = time.time()

    # Lazy import -- requires Docker SDK (only needed for live ingestion)
    from Node_Creation.autonet_core import discover_nodes, get_telemetry

    # -- Step 1: Ingestion ---------------------------------------------------
    print("[ingest] Step 1/5  Discovering Docker containers ...")
    nodes = discover_nodes()
    if not nodes:
        raise RuntimeError(
            "No spine/leaf containers found. "
            "Is Docker running and is the docker-compose stack up?"
        )

    print("[ingest] Step 2/5  Polling 16D telemetry via docker exec ...")
    telemetry = get_telemetry(nodes)
    if not telemetry:
        raise RuntimeError(
            "Telemetry collection returned no results. "
            "Check Docker container connectivity."
        )

    # If attack telemetry is provided, merge it over live telemetry
    mode_label = "LIVE"
    if attack_telemetry:
        mode_label = "ATTACK SIMULATION"
        # Build a lookup of attack overrides by name
        attack_map = {e['name']: e for e in attack_telemetry if isinstance(e, dict)}
        merged = []
        for entry in telemetry:
            name = entry[0]
            if name in attack_map:
                # Use attack-simulated features for this node
                merged.append(attack_map[name])
            else:
                merged.append(entry)
        telemetry = merged

    # Canonical ordered list of node names
    node_names = []
    for entry in telemetry:
        if isinstance(entry, dict):
            node_names.append(entry['name'])
        else:
            node_names.append(entry[0])

    # -- Step 2: Bipartite graph ---------------------------------------------
    print("[ingest] Step 3/5  Building bipartite spine-leaf graph ...")
    G = _build_bipartite_graph(node_names)

    spines = [n for n in node_names if "spine" in n.lower()]
    leaves = [n for n in node_names if "leaf" in n.lower()]
    expected_edges = len(spines) * len(leaves)
    assert G.number_of_edges() == expected_edges, (
        "Edge count mismatch: got %d, expected %d (|S|=%d x |L|=%d)"
        % (G.number_of_edges(), expected_edges, len(spines), len(leaves))
    )

    # -- Step 3: Build + normalise features ----------------------------------
    print("[ingest] Step 4/5  Building and normalising 16D features ...")
    features, labels = _build_features(telemetry, node_names)

    # -- Step 4: PyG conversion ----------------------------------------------
    print("[ingest] Step 5/5  Converting to PyTorch Geometric Data ...")
    data = _to_pyg_data(G, features, labels)

    # -- Save graph ----------------------------------------------------------
    save_path = os.path.join(output_dir, OUTPUT_FILE)
    torch.save(data, save_path)

    # -- Save attack log if provided -----------------------------------------
    if attack_log is not None:
        import json
        from datetime import datetime, timezone
        log_path = os.path.join(output_dir, "attack_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({
                "simulation_timestamp": datetime.now(timezone.utc).isoformat(),
                "attacks": attack_log,
                "total_attacks": len(attack_log),
            }, f, indent=2)
        print("[ingest] Saved attack log : %s" % log_path)

    elapsed = round(time.time() - start, 2)

    # -- Summary -------------------------------------------------------------
    print("")
    print("[ingest] ---- Live Ingestion Complete ----")
    print("[ingest] Mode      : %s" % mode_label)
    print("[ingest] Nodes     : %d  (%d spine, %d leaf)"
          % (data.num_nodes, len(spines), len(leaves)))
    print("[ingest] Edges     : %d undirected  (%d directed in PyG)"
          % (G.number_of_edges(), data.edge_index.shape[1]))
    print("[ingest] Features  : %s  (16D real telemetry, z-normalised)"
          % (tuple(data.x.shape),))
    print("[ingest] Saved     : %s" % save_path)
    print("[ingest] Elapsed   : %s s" % elapsed)

    return data


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ingest()
