"""
generate_topology.py
====================
Generates a synthetic network topology (50 nodes, 16-dim features)
with one injected anomaly node, and exports it as a PyTorch Geometric
Data object.

Requirements:
    pip install networkx numpy torch torch-geometric
"""

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# ──────────────────────────── Configuration ────────────────────────────
NUM_NODES = 50
FEATURE_DIM = 16
ANOMALY_NODE_ID = 0  # the node we'll make anomalous
RANDOM_SEED = 42

# Normal feature distribution
NORMAL_MEAN = 0.0
NORMAL_STD = 1.0

# Anomaly feature distribution – shifted far from normal
ANOMALY_MEAN = 10.0
ANOMALY_STD = 2.0

# ──────────────────────────── Reproducibility ──────────────────────────
np.random.seed(RANDOM_SEED)


def build_topology(num_nodes: int) -> nx.Graph:
    """
    Create a Barabási–Albert scale-free graph, which mimics real-world
    network topologies (hubs, power-law degree distribution).
    Each new node attaches to m=3 existing nodes.
    """
    G = nx.barabasi_albert_graph(n=num_nodes, m=3, seed=RANDOM_SEED)
    return G


def assign_features(G: nx.Graph, anomaly_id: int) -> np.ndarray:
    """
    Assign a 16-dimensional feature vector to every node.
    Normal nodes  ← N(0, 1)
    Anomaly node  ← N(10, 2)   (clearly separable)
    """
    num_nodes = G.number_of_nodes()
    features = np.random.normal(NORMAL_MEAN, NORMAL_STD,
                                size=(num_nodes, FEATURE_DIM))

    # Overwrite the anomaly node's features
    features[anomaly_id] = np.random.normal(ANOMALY_MEAN, ANOMALY_STD,
                                            size=(FEATURE_DIM,))
    return features


def create_labels(num_nodes: int, anomaly_id: int) -> np.ndarray:
    """
    Binary labels: 0 = normal, 1 = anomaly.
    """
    labels = np.zeros(num_nodes, dtype=np.int64)
    labels[anomaly_id] = 1
    return labels


def to_pyg_data(G: nx.Graph, features: np.ndarray,
                labels: np.ndarray) -> Data:
    """
    Convert the NetworkX graph + feature/label arrays into a
    torch_geometric.data.Data object.
    """
    # Edge index from NetworkX (undirected → both directions)
    pyg = from_networkx(G)

    # Attach node features and labels
    pyg.x = torch.tensor(features, dtype=torch.float32)
    pyg.y = torch.tensor(labels, dtype=torch.long)

    # Boolean masks (useful for train/test splits later)
    pyg.train_mask = torch.ones(pyg.num_nodes, dtype=torch.bool)
    pyg.anomaly_mask = pyg.y.bool()

    return pyg


def main():
    # 1. Build topology
    G = build_topology(NUM_NODES)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 2. Assign features
    features = assign_features(G, anomaly_id=ANOMALY_NODE_ID)

    # 3. Create labels
    labels = create_labels(NUM_NODES, anomaly_id=ANOMALY_NODE_ID)

    # 4. Convert to PyG Data
    data = to_pyg_data(G, features, labels)

    # ── Summary ──
    print(f"\nPyG Data object:\n{data}")
    print(f"\nFeature matrix shape : {data.x.shape}")
    print(f"Edge index shape     : {data.edge_index.shape}")
    print(f"Labels               : {data.y}")
    print(f"Anomaly node(s)      : {data.anomaly_mask.nonzero(as_tuple=True)[0].tolist()}")

    # ── Quick sanity check ──
    normal_mean = data.x[~data.anomaly_mask].mean().item()
    anomaly_mean = data.x[data.anomaly_mask].mean().item()
    print(f"\nMean feature value (normal nodes) : {normal_mean:+.4f}")
    print(f"Mean feature value (anomaly node) : {anomaly_mean:+.4f}")

    # 5. Save to disk
    output_path = "synthetic_graph.pt"
    torch.save(data, output_path)
    print(f"\n✔ Saved PyG Data → {output_path}")


if __name__ == "__main__":
    main()
