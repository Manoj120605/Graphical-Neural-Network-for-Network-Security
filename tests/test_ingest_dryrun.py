"""
Offline dry-run test for data/ingest_docker.py internal helpers.
Validates graph construction, feature building, and PyG conversion
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

from data.ingest_docker import _build_bipartite_graph, _build_features, _to_pyg_data

# ---- Mock data (simulates 4 spine + 6 leaf for speed) ---------------------
MOCK_NAMES = [
    "codetocareear-spine-1", "codetocareear-spine-2",
    "codetocareear-spine-3", "codetocareear-spine-4",
    "codetocareear-leaf-1", "codetocareear-leaf-2",
    "codetocareear-leaf-3", "codetocareear-leaf-4",
    "codetocareear-leaf-5", "codetocareear-leaf-6",
]

# Simulate 16D telemetry: leaf-3 has distinctly different traffic pattern
MOCK_TELEMETRY = []
for name in MOCK_NAMES:
    if name == "codetocareear-leaf-3":
        # Anomalous node: high traffic, high CPU, SSH drift
        feats = [50000.0, 80000.0, 5.0, 0.0, 3.0,
                 0.95, 0.88, 120.0, 8.0, 5.0,
                 0.0, 0.0, 3.0, 10.0, 0.0, 1.0]
    else:
        # Normal nodes: low traffic, low CPU, no drift
        feats = [1000.0, 1500.0, 0.0, 0.0, 0.0,
                 0.05, 0.30, 10.0, 0.0, 0.0,
                 0.0, 0.0, 1.0, 4.0, 0.0, 0.0]
    MOCK_TELEMETRY.append([name, feats])

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


def test_feature_building():
    features, labels = _build_features(MOCK_TELEMETRY, MOCK_NAMES)

    # Shape must be (N, 16)
    assert features.shape == (TOTAL, 16), \
        f"Expected shape ({TOTAL}, 16), got {features.shape}"

    # After z-score normalisation, the anomalous node (leaf-3)
    # should have a distinctly different feature vector from the rest
    leaf3_idx = MOCK_NAMES.index("codetocareear-leaf-3")
    leaf3_feats = features[leaf3_idx]

    # At least some features of the anomalous node should be outliers
    # (z-score magnitude > 1.0 for features that have variance)
    outlier_dims = sum(1 for f in leaf3_feats if abs(f) > 1.0)
    assert outlier_dims > 0, \
        "Anomalous node should have at least some outlier features after z-score"

    # Live mode: all labels should be 0 (no ground truth)
    assert labels.sum() == 0, \
        f"Expected all labels to be 0 (live mode), got sum={labels.sum()}"

    print("[PASS] Feature building: shape=%s, outlier_dims=%d, labels all-zero (live mode)"
          % (features.shape, outlier_dims))


def test_pyg_conversion():
    G = _build_bipartite_graph(MOCK_NAMES)
    features, labels = _build_features(MOCK_TELEMETRY, MOCK_NAMES)
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

    # is_live flag
    assert getattr(data, "is_live", False), "data.is_live should be True"

    print("[PASS] PyG conversion: x=%s, y=%s, edges=%d, is_live=True"
          % (tuple(data.x.shape), tuple(data.y.shape), data.edge_index.shape[1]))


if __name__ == "__main__":
    print("=" * 55)
    print("  Dry-Run Validation: data/ingest_docker.py helpers")
    print("=" * 55)
    print()

    test_bipartite_graph()
    test_feature_building()
    test_pyg_conversion()

    print()
    print("All tests passed.")
