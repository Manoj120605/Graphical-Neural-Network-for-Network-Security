#!/usr/bin/env python
"""
run.py -- AutoNet-GNN end-to-end pipeline
==========================================
Execution order (per SYSTEM_DESIGN Section 11):
    1. Generate network graph  +  inject anomaly
    2. Convert to PyTorch Geometric  (integrated in step 1)
    3. Run GNN model  ->  node embeddings
    4. Compute anomaly scores  +  threshold  +  alerts
    5. Visualize anomalies
"""

from data import generate_topology
from detection import detect
from visualization import visualize


def main():
    print("=" * 55)
    print("  AutoNet-GNN  -  Network Anomaly Detection Pipeline")
    print("=" * 55)

    # -- Step 1-2: Generate graph + convert to PyG --
    print("")
    print(">> Step 1  Generate network topology")
    data = generate_topology.generate(output_dir="syntheticdata")

    # -- Step 3-4: GNN embeddings + anomaly scoring + alerts --
    print("")
    print(">> Step 2  Detect anomalies (GraphSAGE + L2 scoring)")
    scores, flagged, threshold = detect(data, output_dir="syntheticdata")

    # -- Step 5: Visualization --
    print("")
    print(">> Step 3  Visualize results")
    out_path = visualize(data, scores, flagged, threshold,
                         output_dir="visualization")

    # -- Summary --
    print("")
    print("=" * 55)
    print("  Pipeline complete")
    print("  Graph     : syntheticdata/synthetic_graph.pt")
    print("  Scores    : syntheticdata/anomaly_scores.csv")
    print("  Alerts    : syntheticdata/alerts.json")
    print("  Plot      : %s" % out_path)
    print("=" * 55)


if __name__ == "__main__":
    main()
