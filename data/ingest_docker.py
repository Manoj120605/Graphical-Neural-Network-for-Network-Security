"""
ingest_docker.py
================
Bridges live Docker-based network simulation with the GraphSAGE
anomaly-detection pipeline.

Pipeline
    1. Ingest   - discover containers & poll telemetry via SSH
    2. Graph    - build strict bipartite spine-leaf topology
    3. Pad      - expand telemetry -> 16D feature vector
    4. Convert  - NetworkX -> PyTorch Geometric Data object
    5. Export   - save to syntheticdata/synthetic_graph.pt

Supports two telemetry modes:
    A) Legacy  : 1D drift_score from get_telemetry()  (zero-padded to 16D)
    B) Enriched: full 16D feature vectors from attack simulation or
                 get_full_telemetry()
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
# (_build_bipartite_graph, _pad_features, _to_pyg_data) can be imported
# and tested without Docker installed.

# ---- Constants ------------------------------------------------------------
FEATURE_DIM = 16          # must match GraphSAGEEncoder(in_dim=16)
DRIFT_INDEX = 0           # position of actual drift score in the 16D vector
OUTPUT_DIR = "syntheticdata"
OUTPUT_FILE = "synthetic_graph.pt"


# ---------------------------------------------------------------------------
# Step 2 helper – bipartite spine-leaf graph
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
# Step 3 helper – pad 1D telemetry to 16D (legacy mode)
# ---------------------------------------------------------------------------
def _pad_features(telemetry: list, node_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Expand telemetry into a 16-dimensional feature vector.

    Supports two input formats:
      A) Legacy : list of [name, drift_score] or {'name': ..., 'drift_score': ...}
                  Score at index 0, zeros elsewhere.
      B) Enriched: list of {'name': ..., 'features': [16 values], 'drift_score': ...}
                   Uses the full 16D feature vector directly.

    Parameters
    ----------
    telemetry : list
        Telemetry entries from get_telemetry(), get_full_telemetry(),
        or attack simulator.
    node_names : list[str]
        Ordered master list of container names (defines row ordering).

    Returns
    -------
    features : np.ndarray, shape (N, 16)
    labels   : np.ndarray, shape (N,)  — 1 if drift_score == 1.0 else 0
    """
    # Normalize telemetry into a consistent dict format
    drift_map: dict[str, float] = {}
    feature_map: dict[str, list[float]] = {}

    for entry in telemetry:
        if isinstance(entry, dict):
            name = entry["name"]
            drift_map[name] = float(entry.get("drift_score", 0.0))
            if "features" in entry and len(entry["features"]) == FEATURE_DIM:
                feature_map[name] = entry["features"]
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            name = entry[0]
            drift_map[name] = float(entry[1])
        else:
            continue

    num_nodes = len(node_names)
    features = np.zeros((num_nodes, FEATURE_DIM), dtype=np.float32)
    labels = np.zeros(num_nodes, dtype=np.int64)

    for idx, name in enumerate(node_names):
        score = drift_map.get(name, 0.0)
        labels[idx] = 1 if score == 1.0 else 0

        if name in feature_map:
            # Enriched mode: use full 16D vector
            features[idx, :] = feature_map[name]
        else:
            # Legacy mode: drift score at index 0, rest zero-padded
            features[idx, DRIFT_INDEX] = score

    return features, labels


# ---------------------------------------------------------------------------
# Step 4 helper – PyG conversion
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
    2. Poll telemetry (SSH config drift or full metrics).
    3. Optionally overlay attack simulation telemetry.
    4. Build bipartite spine-leaf graph.
    5. Pad features to 16D.
    6. Convert to PyG ``Data`` and save to disk.

    Parameters
    ----------
    output_dir : str
        Directory to save outputs (default: "syntheticdata").
    attack_telemetry : list[dict] | None
        If provided, uses attack simulator telemetry instead of polling
        containers for telemetry. Each dict must have 'name', 'features',
        and 'drift_score' keys.
    attack_log : list[dict] | None
        If provided, saves the attack log alongside graph data.

    Returns
    -------
    data : torch_geometric.data.Data
    """
    os.makedirs(output_dir, exist_ok=True)
    start = time.time()

    # Lazy import — requires Docker SDK (only needed for live ingestion)
    from Node_Creation.autonet_core import discover_nodes, get_telemetry

    # -- Step 1: Ingestion ---------------------------------------------------
    print("[ingest] Step 1/5  Discovering Docker containers ...")
    nodes = discover_nodes()
    if not nodes:
        raise RuntimeError(
            "No spine/leaf containers found. "
            "Is Docker running and is the docker-compose stack up?"
        )

    # -- Step 2: Telemetry ---------------------------------------------------
    if attack_telemetry is not None:
        # Attack simulation mode: use pre-generated attack telemetry
        print("[ingest] Step 2/5  Using attack simulation telemetry ...")
        telemetry = attack_telemetry
        node_names = [entry["name"] for entry in telemetry]
    else:
        # Normal mode: poll containers directly
        print("[ingest] Step 2/5  Polling telemetry via SSH ...")
        telemetry = get_telemetry(nodes)
        if not telemetry:
            raise RuntimeError(
                "Telemetry collection returned no results. "
                "Check SSH connectivity to the containers."
            )
        # Extract node names from telemetry
        node_names = [
            entry["name"] if isinstance(entry, dict) else entry[0]
            for entry in telemetry
        ]

    # -- Step 3: Bipartite graph ---------------------------------------------
    print("[ingest] Step 3/5  Building bipartite spine-leaf graph ...")
    G = _build_bipartite_graph(node_names)

    spines = [n for n in node_names if "spine" in n.lower()]
    leaves = [n for n in node_names if "leaf" in n.lower()]
    expected_edges = len(spines) * len(leaves)
    assert G.number_of_edges() == expected_edges, (
        f"Edge count mismatch: got {G.number_of_edges()}, "
        f"expected {expected_edges} (|S|={len(spines)} x |L|={len(leaves)})"
    )

    # -- Step 4: Pad features ------------------------------------------------
    print("[ingest] Step 4/5  Processing features to %dD ..." % FEATURE_DIM)
    features, labels = _pad_features(telemetry, node_names)

    # -- Step 5: PyG conversion ----------------------------------------------
    print("[ingest] Step 5/5  Converting to PyTorch Geometric Data ...")
    data = _to_pyg_data(G, features, labels)

    # -- Save graph ----------------------------------------------------------
    save_path = os.path.join(output_dir, OUTPUT_FILE)
    torch.save(data, save_path)

    # -- Save attack log if provided -----------------------------------------
    if attack_log is not None:
        import json
        log_path = os.path.join(output_dir, "attack_log.json")
        from datetime import datetime, timezone
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({
                "simulation_timestamp": datetime.now(timezone.utc).isoformat(),
                "attacks": attack_log,
                "total_attacks": len(attack_log),
            }, f, indent=2)
        print("[ingest] Saved attack log : %s" % log_path)

    elapsed = round(time.time() - start, 2)

    # -- Summary -------------------------------------------------------------
    anomaly_count = int(labels.sum())
    enriched = any(
        isinstance(e, dict) and "features" in e for e in telemetry
    )
    mode_label = "enriched 16D" if enriched else "1D drift (zero-padded)"

    print("")
    print("[ingest] ---- Live Ingestion Complete ----")
    print("[ingest] Mode      : %s" % mode_label)
    print("[ingest] Nodes     : %d  (%d spine, %d leaf)"
          % (data.num_nodes, len(spines), len(leaves)))
    print("[ingest] Edges     : %d undirected  (%d directed in PyG)"
          % (G.number_of_edges(), data.edge_index.shape[1]))
    print("[ingest] Features  : %s  (mode: %s)"
          % (tuple(data.x.shape), mode_label))
    print("[ingest] Anomalies : %d / %d nodes (drift_score == 1.0)"
          % (anomaly_count, data.num_nodes))
    print("[ingest] Saved     : %s" % save_path)
    print("[ingest] Elapsed   : %s s" % elapsed)

    return data


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ingest()
