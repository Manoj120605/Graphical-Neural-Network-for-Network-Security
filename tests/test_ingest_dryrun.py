"""
Offline dry-run test for data/ingest_docker.py internal helpers.
Validates graph construction, feature padding, and PyG conversion
WITHOUT requiring Docker or SSH connectivity.
"""
import os
import sys

# Ensure project root is on the path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# We need to make the project root the one that matches the repo
os.chdir(PROJECT_ROOT)

from data.ingest_docker import _build_bipartite_graph, _pad_features, _to_pyg_data

# ---- Mock data (simulates 4 spine + 6 leaf for speed) ---------------------
MOCK_NAMES = [
    "codetocareear-spine-1", "codetocareear-spine-2",
    "codetocareear-spine-3", "codetocareear-spine-4",
    "codetocareear-leaf-1", "codetocareear-leaf-2",
    "codetocareear-leaf-3", "codetocareear-leaf-4",
    "codetocareear-leaf-5", "codetocareear-leaf-6",
]

MOCK_TELEMETRY = [
    [name, 1.0 if name == "codetocareear-leaf-3" else 0.0]
    for name in MOCK_NAMES
]

NUM_SPINE = 4
NUM_LEAF = 6
TOTAL = NUM_SPINE + NUM_LEAF


def test_bipartite_graph():
    G = _build_bipartite_graph(MOCK_NAMES)

    # Correct number of nodes
    assert G.number_of_nodes() == TOTAL, \
        f"Expected {TOTAL} nodes, got {G.number_of_nodes()}"

    # Correct edge count: |spine| x |leaf|
    expected_edges = NUM_SPINE * NUM_LEAF
    assert G.number_of_edges() == expected_edges, \
        f"Expected {expected_edges} edges, got {G.number_of_edges()}"

    # No intra-tier edges
    for u, v in G.edges():
        u_role = G.nodes[u]["role"]
        v_role = G.nodes[v]["role"]
        assert u_role != v_role, \
            f"Intra-tier edge found: {u} ({u_role}) -- {v} ({v_role})"

    print("[PASS] Bipartite graph: %d nodes, %d edges, no intra-tier edges"
          % (TOTAL, expected_edges))


def test_feature_padding():
    features, labels = _pad_features(MOCK_TELEMETRY, MOCK_NAMES)

    # Shape must be (N, 16)
    assert features.shape == (TOTAL, 16), \
        f"Expected shape ({TOTAL}, 16), got {features.shape}"

    # Drift scores at index 0
    for idx, name in enumerate(MOCK_NAMES):
        expected_score = 1.0 if name == "codetocareear-leaf-3" else 0.0
        assert features[idx, 0] == expected_score, \
            f"Node {name}: expected drift={expected_score}, got {features[idx, 0]}"

    # Indices 1-15 must be zero
    assert (features[:, 1:] == 0.0).all(), \
        "Non-zero values found in padded dimensions 1-15"

    # Labels: 1 for anomaly, 0 for normal
    anomaly_count = int(labels.sum())
    assert anomaly_count == 1, \
        f"Expected 1 anomaly, got {anomaly_count}"

    print("[PASS] Feature padding: shape=%s, anomalies=%d, padding correct"
          % (features.shape, anomaly_count))


def test_pyg_conversion():
    G = _build_bipartite_graph(MOCK_NAMES)
    features, labels = _pad_features(MOCK_TELEMETRY, MOCK_NAMES)
    data = _to_pyg_data(G, features, labels)

    # data.x shape
    assert data.x.shape == (TOTAL, 16), \
        f"data.x shape mismatch: {data.x.shape}"
    assert data.x.dtype.__str__() == "torch.float32", \
        f"data.x dtype mismatch: {data.x.dtype}"

    # data.y shape
    assert data.y.shape == (TOTAL,), \
        f"data.y shape mismatch: {data.y.shape}"
    assert data.y.dtype.__str__() == "torch.int64", \
        f"data.y dtype mismatch: {data.y.dtype}"

    # edge_index: undirected -> 2x edges in each direction
    expected_directed = NUM_SPINE * NUM_LEAF * 2
    assert data.edge_index.shape[1] == expected_directed, \
        f"Expected {expected_directed} directed edges, got {data.edge_index.shape[1]}"

    # Masks
    assert data.train_mask.all(), "train_mask should be all True"
    assert data.anomaly_mask.sum().item() == 1, \
        f"anomaly_mask should flag exactly 1 node"

    print("[PASS] PyG conversion: x=%s, y=%s, edges=%d, masks OK"
          % (tuple(data.x.shape), tuple(data.y.shape), data.edge_index.shape[1]))


if __name__ == "__main__":
    print("=" * 55)
    print("  Dry-Run Validation: data/ingest_docker.py helpers")
    print("=" * 55)
    print()

    test_bipartite_graph()
    test_feature_padding()
    test_pyg_conversion()

    print()
    print("All tests passed.")
