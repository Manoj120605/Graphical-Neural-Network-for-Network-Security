"""
ingest_docker.py
================
Bridges live Docker-based network simulation with the GraphSAGE
anomaly-detection pipeline.

Pipeline
    1. Ingest   - discover containers & poll telemetry via SSH
    2. Graph    - build strict bipartite spine-leaf topology
    3. Pad      - expand 1D drift score -> 16D feature vector
    4. Convert  - NetworkX -> PyTorch Geometric Data object
    5. Export   - save to syntheticdata/synthetic_graph.pt
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

from Node_Creation.autonet_core import discover_nodes, get_telemetry

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
# Step 3 helper – pad 1D telemetry to 16D
# ---------------------------------------------------------------------------
def _pad_features(telemetry: list[list], node_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Expand the scalar ``telnet_drift_score`` into a 16-dimensional feature
    vector (score at index 0, zeros elsewhere).  Also derive binary labels.

    Parameters
    ----------
    telemetry : list[list]
        Each element is ``[node_name, drift_score]`` from ``get_telemetry()``.
    node_names : list[str]
        Ordered master list of container names (defines row ordering).

    Returns
    -------
    features : np.ndarray, shape (N, 16)
    labels   : np.ndarray, shape (N,)  — 1 if drift_score == 1.0 else 0
    """
    # Map name -> drift_score for O(1) lookup
    drift_map: dict[str, float] = {entry[0]: float(entry[1]) for entry in telemetry}

    num_nodes = len(node_names)
    features = np.zeros((num_nodes, FEATURE_DIM), dtype=np.float32)
    labels = np.zeros(num_nodes, dtype=np.int64)

    for idx, name in enumerate(node_names):
        score = drift_map.get(name, 0.0)
        features[idx, DRIFT_INDEX] = score
        labels[idx] = 1 if score == 1.0 else 0

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
def ingest(output_dir: str = OUTPUT_DIR) -> Data:
    """
    End-to-end live-ingestion pipeline.

    1. Discover Docker containers (spine/leaf).
    2. SSH into each container and poll telnet-drift telemetry.
    3. Build bipartite spine-leaf graph.
    4. Pad 1D features to 16D.
    5. Convert to PyG ``Data`` and save to disk.

    Returns
    -------
    data : torch_geometric.data.Data
    """
    os.makedirs(output_dir, exist_ok=True)
    start = time.time()

    # -- Step 1: Ingestion ---------------------------------------------------
    print("[ingest] Step 1/5  Discovering Docker containers ...")
    nodes = discover_nodes()
    if not nodes:
        raise RuntimeError(
            "No spine/leaf containers found. "
            "Is Docker running and is the docker-compose stack up?"
        )

    print("[ingest] Step 1/5  Polling telemetry via SSH ...")
    telemetry = get_telemetry(nodes)
    if not telemetry:
        raise RuntimeError(
            "Telemetry collection returned no results. "
            "Check SSH connectivity to the containers."
        )

    # Canonical ordered list of node names (spines first for readability)
    node_names = [entry[0] for entry in telemetry]

    # -- Step 2: Bipartite graph ---------------------------------------------
    print("[ingest] Step 2/5  Building bipartite spine-leaf graph ...")
    G = _build_bipartite_graph(node_names)

    spines = [n for n in node_names if "spine" in n.lower()]
    leaves = [n for n in node_names if "leaf" in n.lower()]
    expected_edges = len(spines) * len(leaves)
    assert G.number_of_edges() == expected_edges, (
        f"Edge count mismatch: got {G.number_of_edges()}, "
        f"expected {expected_edges} (|S|={len(spines)} x |L|={len(leaves)})"
    )

    # -- Step 3: Pad features ------------------------------------------------
    print("[ingest] Step 3/5  Padding features to %dD ..." % FEATURE_DIM)
    features, labels = _pad_features(telemetry, node_names)

    # -- Step 4: PyG conversion ----------------------------------------------
    print("[ingest] Step 4/5  Converting to PyTorch Geometric Data ...")
    data = _to_pyg_data(G, features, labels)

    # -- Step 5: Export ------------------------------------------------------
    save_path = os.path.join(output_dir, OUTPUT_FILE)
    torch.save(data, save_path)

    elapsed = round(time.time() - start, 2)

    # -- Summary -------------------------------------------------------------
    anomaly_count = int(labels.sum())
    print("")
    print("[ingest] ---- Live Ingestion Complete ----")
    print("[ingest] Nodes     : %d  (%d spine, %d leaf)"
          % (data.num_nodes, len(spines), len(leaves)))
    print("[ingest] Edges     : %d undirected  (%d directed in PyG)"
          % (G.number_of_edges(), data.edge_index.shape[1]))
    print("[ingest] Features  : %s  (drift at idx %d, rest zero-padded)"
          % (tuple(data.x.shape), DRIFT_INDEX))
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
