"""
generate_topology.py
====================
Builds a synthetic Barabási–Albert network topology (50 nodes, 16-dim
telemetry features), injects one anomaly node, and exports the result
as a PyTorch Geometric Data object.
"""

import os
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# ──────────────────────────── Configuration ────────────────────────────
NUM_NODES = 50
FEATURE_DIM = 16
ANOMALY_NODE_ID = 0
RANDOM_SEED = 42

# Feature names matching SYSTEM_DESIGN §4
FEATURE_NAMES = [
    "traffic_in", "traffic_out", "packet_loss", "latency",
    "crc_errors", "cpu_usage", "memory_usage", "connection_count",
    "interface_errors", "dropped_packets", "jitter", "link_utilization",
    "route_changes", "neighbor_count", "retransmissions", "queue_depth",
]

# Normal feature distribution
NORMAL_MEAN = 0.0
NORMAL_STD = 1.0

# Anomaly feature distribution — shifted far from normal
ANOMALY_MEAN = 10.0
ANOMALY_STD = 2.0

# ──────────────────────────── Reproducibility ──────────────────────────
np.random.seed(RANDOM_SEED)


def build_topology(num_nodes: int = NUM_NODES) -> nx.Graph:
    """
    Create a Barabási–Albert scale-free graph, which mimics real-world
    network topologies (hubs, power-law degree distribution).
    Each new node attaches to m=3 existing nodes.
    """
    G = nx.barabasi_albert_graph(n=num_nodes, m=3, seed=RANDOM_SEED)
    return G


def assign_features(G: nx.Graph,
                    anomaly_id: int = ANOMALY_NODE_ID) -> np.ndarray:
    """
    Assign a 16-dimensional feature vector to every node.
        Normal nodes  ← N(0, 1)
        Anomaly node  ← N(10, 2)   (clearly separable)
    """
    num_nodes = G.number_of_nodes()
    features = np.random.normal(NORMAL_MEAN, NORMAL_STD,
                                size=(num_nodes, FEATURE_DIM))
    features[anomaly_id] = np.random.normal(ANOMALY_MEAN, ANOMALY_STD,
                                            size=(FEATURE_DIM,))
    return features


def create_labels(num_nodes: int = NUM_NODES,
                  anomaly_id: int = ANOMALY_NODE_ID) -> np.ndarray:
    """Binary labels: 0 = normal, 1 = anomaly."""
    labels = np.zeros(num_nodes, dtype=np.int64)
    labels[anomaly_id] = 1
    return labels


def to_pyg_data(G: nx.Graph, features: np.ndarray,
                labels: np.ndarray) -> Data:
    """
    Convert the NetworkX graph + feature/label arrays into a
    torch_geometric.data.Data object.
    """
    pyg = from_networkx(G)
    pyg.x = torch.tensor(features, dtype=torch.float32)
    pyg.y = torch.tensor(labels, dtype=torch.long)
    pyg.train_mask = torch.ones(pyg.num_nodes, dtype=torch.bool)
    pyg.anomaly_mask = pyg.y.bool()
    return pyg


def generate(output_dir: str = "syntheticdata") -> Data:
    """
    Full generation pipeline — build graph, assign features, inject
    anomaly, convert to PyG, and save to disk.

    Returns the PyG Data object.
    """
    os.makedirs(output_dir, exist_ok=True)

    G = build_topology(NUM_NODES)
    features = assign_features(G, anomaly_id=ANOMALY_NODE_ID)
    labels = create_labels(NUM_NODES, anomaly_id=ANOMALY_NODE_ID)
    data = to_pyg_data(G, features, labels)

    save_path = os.path.join(output_dir, "synthetic_graph.pt")
    torch.save(data, save_path)

    print(f"[data] Graph : {data.num_nodes} nodes, "
          f"{data.edge_index.shape[1]} directed edges")
    print(f"[data] Features : {data.x.shape}  "
          f"(normal μ={NORMAL_MEAN}, anomaly μ={ANOMALY_MEAN})")
    print(f"[data] Anomaly  : node {ANOMALY_NODE_ID}")
    print(f"[data] Saved    → {save_path}")

    return data


if __name__ == "__main__":
    generate()
