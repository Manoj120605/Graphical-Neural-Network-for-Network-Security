"""
visualize_graph.py
==================
Draws the network topology with anomaly-flagged nodes highlighted in red
and normal nodes in green. Saves the plot to  graph_anomaly_map.png.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
from model import GraphSAGEEncoder
from detect_anomalies import compute_anomaly_scores


def main():
    # ── Load graph ──
    data = torch.load("synthetic_graph.pt", weights_only=False)

    # ── Generate embeddings & anomaly scores ──
    model = GraphSAGEEncoder(in_dim=16, hidden_dim=32, out_dim=16)
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)
    scores = compute_anomaly_scores(embeddings, data.edge_index)

    # ── Threshold (μ + 2σ) ──
    threshold = scores.mean().item() + 2 * scores.std().item()

    # ── Rebuild NetworkX graph from edge_index ──
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    edges = data.edge_index.t().tolist()
    G.add_edges_from(edges)

    # ── Node colours ──
    node_colors = [
        "#E74C3C" if scores[n].item() > threshold else "#2ECC71"
        for n in G.nodes()
    ]

    # ── Node sizes (scale by anomaly score for emphasis) ──
    max_score = scores.max().item()
    node_sizes = [
        300 + 700 * (scores[n].item() / max_score) for n in G.nodes()
    ]

    # ── Layout ──
    pos = nx.spring_layout(G, seed=42, k=0.55)

    # ── Draw ──
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#1A1A2E")

    # Edges
    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color="#3A3A5C", width=0.8, alpha=0.6)

    # Nodes
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors,
                           node_size=node_sizes,
                           edgecolors="#FFFFFF", linewidths=0.8)

    # Labels
    nx.draw_networkx_labels(G, pos, ax=ax,
                            font_size=7, font_color="#FFFFFF",
                            font_weight="bold")

    # ── Annotate flagged nodes ──
    flagged = [n for n in G.nodes() if scores[n].item() > threshold]
    for n in flagged:
        x, y = pos[n]
        ax.annotate(f"score={scores[n].item():.2f}",
                    xy=(x, y), xytext=(12, 12),
                    textcoords="offset points",
                    fontsize=8, color="#E74C3C", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#E74C3C",
                                    lw=1.2))

    # ── Legend ──
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="o", color="#1A1A2E", markerfacecolor="#E74C3C",
               markersize=12, label=f"Anomalous  (score > {threshold:.2f})"),
        Line2D([0], [0], marker="o", color="#1A1A2E", markerfacecolor="#2ECC71",
               markersize=12, label="Normal"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=10,
              facecolor="#16213E", edgecolor="#3A3A5C", labelcolor="#FFFFFF")

    ax.set_title("Network Topology — Anomaly Detection",
                 fontsize=16, fontweight="bold", color="#FFFFFF", pad=20)
    ax.axis("off")
    plt.tight_layout()

    # ── Save ──
    output = "graph_anomaly_map.png"
    fig.savefig(output, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✔ Saved visualisation → {output}")
    print(f"  Threshold : {threshold:.4f}")
    print(f"  Flagged   : {flagged}")


if __name__ == "__main__":
    main()
