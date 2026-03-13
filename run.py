#!/usr/bin/env python
"""
run.py — AutoNet-GNN end-to-end pipeline
=========================================
Execution order (per SYSTEM_DESIGN §11):
    1. Generate network graph  +  inject anomaly
    2. Convert to PyTorch Geometric
    3. Run GNN model  →  node embeddings
    4. Compute anomaly scores  +  threshold
    5. Visualize anomalies
"""

from data import generate_topology
from detection import detect
from visualization import visualize


def main():
    print("=" * 55)
    print("  AutoNet-GNN  ·  Network Anomaly Detection Pipeline")
    print("=" * 55)

    # ── Step 1-2: Generate graph + convert to PyG ──
    print("\n▶ Step 1  Generate network topology")
    data = generate_topology.generate(output_dir="syntheticdata")

    # ── Step 3-4: GNN embeddings + anomaly scoring ──
    print("\n▶ Step 2  Detect anomalies (GraphSAGE + L2 scoring)")
    scores, flagged, threshold = detect(data, output_dir="syntheticdata")

    # ── Step 5: Visualization ──
    print("\n▶ Step 3  Visualize results")
    out_path = visualize(data, scores, flagged, threshold,
                         output_dir="visualization")

    # ── Done ──
    print("\n" + "=" * 55)
    print("  Pipeline complete")
    print(f"  Graph     : syntheticdata/synthetic_graph.pt")
    print(f"  Scores    : syntheticdata/anomaly_scores.csv")
    print(f"  Plot      : {out_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
