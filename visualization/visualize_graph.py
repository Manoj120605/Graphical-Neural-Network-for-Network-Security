"""
visualize_graph.py
==================
Draws the network topology with anomaly-flagged nodes highlighted
in red and normal nodes in green.
Saves to visualization/graph_anomaly_map.png.
"""

import os
import matplotlib.pyplot as plt
import networkx as nx
import torch
from matplotlib.lines import Line2D
from torch_geometric.data import Data


def visualize(data: Data,
              scores: torch.Tensor,
              flagged: list[int],
              threshold: float,
              output_dir: str = "visualization") -> str:
    """
    Render the graph with colour-coded nodes and save as PNG.

    Returns:
        Path to the saved image.
    """
    os.makedirs(output_dir, exist_ok=True)

    # -- Rebuild NetworkX graph --
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(data.edge_index.t().tolist())

    flagged_set = set(flagged)

    # -- Colours: red = anomalous, green = normal --
    node_colors = [
        "#E74C3C" if n in flagged_set else "#2ECC71"
        for n in G.nodes()
    ]

    max_score = scores.max().item() or 1.0
    node_sizes = [
        300 + 700 * (scores[n].item() / max_score) for n in G.nodes()
    ]

    pos = nx.spring_layout(G, seed=42, k=0.55)

    # -- Draw --
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#1A1A2E")

    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color="#3A3A5C", width=0.8, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors,
                           node_size=node_sizes,
                           edgecolors="#FFFFFF", linewidths=0.8)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            font_size=7, font_color="#FFFFFF",
                            font_weight="bold")

    # -- Annotate flagged nodes --
    for n in flagged:
        x, y = pos[n]
        ax.annotate("score=%.2f" % scores[n].item(),
                    xy=(x, y), xytext=(12, 12),
                    textcoords="offset points",
                    fontsize=8, color="#E74C3C", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#E74C3C",
                                    lw=1.2))

    # -- Legend --
    legend = [
        Line2D([0], [0], marker="o", color="#1A1A2E",
               markerfacecolor="#E74C3C", markersize=12,
               label="Anomalous  (score > %.2f)" % threshold),
        Line2D([0], [0], marker="o", color="#1A1A2E",
               markerfacecolor="#2ECC71", markersize=12,
               label="Normal"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=10,
              facecolor="#16213E", edgecolor="#3A3A5C", labelcolor="#FFFFFF")

    ax.set_title("Network Topology -- Anomaly Detection",
                 fontsize=16, fontweight="bold", color="#FFFFFF", pad=20)
    ax.axis("off")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "graph_anomaly_map.png")
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)

    print("")
    print("[viz] Saved : %s" % out_path)
    return out_path
