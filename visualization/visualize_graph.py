"""
visualize_graph.py
==================
Two visualization outputs:
  1. Static PNG  -- visualize()            (matplotlib)
  2. Interactive -- visualize_interactive() (vis.js SOC dashboard)

The interactive dashboard features:
  - Glassmorphism panels with glowing borders
  - Animated gradient header
  - Attack timeline with MITRE ATT&CK tags
  - Dual-plane verification panel
  - Feature vector heatmap inspector
  - Neural confidence gauge
"""

import os
import json
import matplotlib.pyplot as plt
import networkx as nx
import torch
from matplotlib.lines import Line2D
from torch_geometric.data import Data


# ──────────────────────── CONSTANTS ────────────────────────────────────
FEATURE_NAMES = [
    "traffic_in", "traffic_out", "packet_loss", "latency",
    "crc_errors", "cpu_usage", "memory_usage", "connection_count",
    "interface_errors", "dropped_packets", "jitter", "link_utilization",
    "route_changes", "neighbor_count", "retransmissions", "queue_depth",
]

DEVICE_TYPES = [
    "Router", "Core Switch", "Access Switch", "Firewall", "Server",
    "Load Balancer", "DNS Server", "Gateway", "AP Controller", "IDS Sensor",
]


# ──────────────────────── STATIC PNG ──────────────────────────────────
def visualize(data: Data,
              scores: torch.Tensor,
              flagged: list[int],
              threshold: float,
              output_dir: str = "visualization") -> str:
    """Render static matplotlib PNG."""
    os.makedirs(output_dir, exist_ok=True)
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(data.edge_index.t().tolist())

    flagged_set = set(flagged)
    node_colors = ["#E74C3C" if n in flagged_set else "#2ECC71" for n in G.nodes()]
    max_score = scores.max().item() or 1.0
    node_sizes = [300 + 700 * (scores[n].item() / max_score) for n in G.nodes()]
    pos = nx.spring_layout(G, seed=42, k=0.55)

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#1A1A2E")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#3A3A5C", width=0.8, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, edgecolors="#FFFFFF", linewidths=0.8)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color="#FFFFFF", font_weight="bold")
    for n in flagged:
        x, y = pos[n]
        ax.annotate("score=%.2f" % scores[n].item(), xy=(x, y), xytext=(12, 12),
                    textcoords="offset points", fontsize=8, color="#E74C3C", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.2))
    legend = [
        Line2D([0], [0], marker="o", color="#1A1A2E", markerfacecolor="#E74C3C",
               markersize=12, label="Anomalous  (score > %.2f)" % threshold),
        Line2D([0], [0], marker="o", color="#1A1A2E", markerfacecolor="#2ECC71",
               markersize=12, label="Normal"),
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
    print("[viz] Static PNG : %s" % out_path)
    return out_path


# ──────────────────── INTERACTIVE SOC DASHBOARD ───────────────────────

def visualize_interactive(data: Data,
                          scores: torch.Tensor,
                          flagged: list[int],
                          threshold: float,
                          output_dir: str = "visualization",
                          classifications: dict = None) -> str:
    """
    Generate a premium SOC-style interactive HTML dashboard
    with glassmorphism, attack timeline, and dual-plane verification.
    """
    os.makedirs(output_dir, exist_ok=True)

    num_nodes = data.num_nodes
    num_edges = data.edge_index.shape[1] // 2
    flagged_set = set(flagged)
    max_score = scores.max().item() or 1.0

    # Classification colors
    cls_colors = {
        "CLEAN": "#2ecc71",
        "FEATURE_POISONED": "#f39c12",
        "STRUCTURE_POISONED": "#9b59b6",
        "DUAL_POISONED": "#e74c3c",
    }

    # ── Build vis.js nodes ──
    vis_nodes = []
    node_meta = {}
    for n in range(num_nodes):
        is_flagged = n in flagged_set
        score = scores[n].item()
        device = DEVICE_TYPES[n % len(DEVICE_TYPES)]

        neighbors = []
        for i in range(data.edge_index.shape[1]):
            if data.edge_index[0, i].item() == n:
                neighbors.append(data.edge_index[1, i].item())

        features = {}
        for j in range(min(len(FEATURE_NAMES), data.x.shape[1])):
            features[FEATURE_NAMES[j]] = round(data.x[n, j].item(), 4)

        # Get classification if available
        cls_info = {}
        if classifications and n in classifications:
            cls_info = classifications[n]

        node_meta[n] = {
            "id": n, "score": round(score, 4),
            "label": int(data.y[n].item()), "is_anomaly": is_flagged,
            "device_type": device, "neighbors": neighbors,
            "features": features,
            "classification": cls_info.get("classification", "UNKNOWN"),
            "confidence": cls_info.get("confidence", 0),
            "severity": cls_info.get("severity", "NORMAL"),
            "top_features": cls_info.get("top_features", []),
        }

        cls = cls_info.get("classification", "CLEAN")
        node_color = cls_colors.get(cls, "#2ecc71") if is_flagged else "#2ecc71"

        if is_flagged:
            size = 28 + 30 * (score / max_score)
            vis_nodes.append({
                "id": n, "label": str(n), "size": size,
                "color": {
                    "background": node_color, "border": "#ffffff",
                    "highlight": {"background": node_color, "border": "#ffffff"},
                    "hover": {"background": node_color, "border": "#ffffff"},
                },
                "shadow": {
                    "enabled": True, "color": node_color + "99",
                    "size": 30, "x": 0, "y": 0,
                },
                "borderWidth": 3,
                "font": {"size": 14, "color": "#ffffff", "bold": {"color": "#ffffff"}},
            })
        else:
            size = 12 + 18 * (score / max_score)
            vis_nodes.append({
                "id": n, "label": str(n), "size": size,
                "color": {
                    "background": "#1a3a5c", "border": "#2a5a8c",
                    "highlight": {"background": "#2a5a8c", "border": "#ffffff"},
                    "hover": {"background": "#2a5a8c", "border": "#ffffff"},
                },
                "shadow": {"enabled": False},
                "borderWidth": 1.5,
                "font": {"size": 11, "color": "#8899aa"},
            })

    # ── Build vis.js edges ──
    vis_edges = []
    seen = set()
    for i in range(data.edge_index.shape[1]):
        s = data.edge_index[0, i].item()
        d = data.edge_index[1, i].item()
        key = (min(s, d), max(s, d))
        if key in seen:
            continue
        seen.add(key)
        if s in flagged_set or d in flagged_set:
            vis_edges.append({"from": s, "to": d,
                              "color": {"color": "#ff4d4d", "opacity": 0.4},
                              "width": 2.0})
        else:
            vis_edges.append({"from": s, "to": d,
                              "color": {"color": "#0d2137", "opacity": 0.35},
                              "width": 0.8})

    # ── Load extra data for dashboard ──
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    alerts_path = os.path.join(_project_root, "syntheticdata", "alerts.json")
    attack_log_path = os.path.join(_project_root, "syntheticdata", "attack_log.json")
    dp_report_path = os.path.join(_project_root, "syntheticdata", "dual_plane_report.json")
    root_cause_path = os.path.join(_project_root, "syntheticdata", "root_cause_report.json")

    alerts_list = []
    if os.path.exists(alerts_path):
        with open(alerts_path, "r", encoding="utf-8") as f:
            alerts_list = json.load(f).get("alerts", [])

    attack_events = []
    if os.path.exists(attack_log_path):
        with open(attack_log_path, "r", encoding="utf-8") as f:
            attack_events = json.load(f).get("attacks", [])

    dp_summary = {}
    if os.path.exists(dp_report_path):
        with open(dp_report_path, "r", encoding="utf-8") as f:
            dp_summary = json.load(f).get("summary", {})

    root_cause_text = ""
    if os.path.exists(root_cause_path):
        with open(root_cause_path, "r", encoding="utf-8") as f:
            rc = json.load(f)
        rc_data = rc.get("analyses", {}).get("root_cause", {})
        root_cause_text = rc_data.get("response", "Analysis pending...")

    has_anomalies = len(flagged) > 0

    # Dual-plane counts
    dp_cls = dp_summary.get("classifications", {})
    n_clean = dp_cls.get("CLEAN", 0)
    n_feat = dp_cls.get("FEATURE_POISONED", 0)
    n_struct = dp_cls.get("STRUCTURE_POISONED", 0)
    n_dual = dp_cls.get("DUAL_POISONED", 0)

    # Neural confidence (avg of non-clean confidence)
    confidence_scores = []
    if classifications:
        for v in classifications.values():
            if v.get("classification") != "CLEAN":
                confidence_scores.append(v.get("confidence", 0))
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

    html = _build_html(
        vis_nodes_json=json.dumps(vis_nodes),
        vis_edges_json=json.dumps(vis_edges),
        node_meta_json=json.dumps(node_meta),
        alerts_json=json.dumps(alerts_list),
        attacks_json=json.dumps(attack_events),
        num_nodes=num_nodes,
        num_edges=num_edges,
        num_flagged=len(flagged),
        threshold=threshold,
        has_anomalies=has_anomalies,
        n_clean=n_clean,
        n_feat=n_feat,
        n_struct=n_struct,
        n_dual=n_dual,
        avg_confidence=avg_confidence,
        root_cause_text=root_cause_text,
    )

    out_path = os.path.join(output_dir, "dashboard.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print("[viz] Dashboard : %s" % out_path)
    return out_path


def _build_html(vis_nodes_json, vis_edges_json, node_meta_json,
                alerts_json, attacks_json,
                num_nodes, num_edges, num_flagged,
                threshold, has_anomalies,
                n_clean, n_feat, n_struct, n_dual,
                avg_confidence, root_cause_text):

    live_color = "#ff4d4d" if has_anomalies else "#2ecc71"
    live_text = "THREATS DETECTED" if has_anomalies else "ALL CLEAR"
    conf_pct = int(avg_confidence * 100)

    # Escape root cause text for JS
    rc_escaped = json.dumps(root_cause_text)

    return (
        """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AutoNet-GNN | SOC Dashboard — Dual Poisoning Defense</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg-deep:#060a10;--bg-card:rgba(12,18,30,0.85);--bg-glass:rgba(20,30,50,0.6);
  --border:rgba(56,97,150,0.25);--border-glow:rgba(88,166,255,0.3);
  --text:#c8d4e0;--text-dim:#5a6a7e;--text-bright:#e8f0fa;
  --accent:#58a6ff;--accent2:#7c3aed;--red:#ef4444;--green:#22c55e;
  --orange:#f59e0b;--purple:#a78bfa;--cyan:#06b6d4;
}
body{font-family:'Inter',system-ui,sans-serif;background:var(--bg-deep);color:var(--text);overflow-x:hidden;height:100vh}

/* ═══ ANIMATED PARTICLE BG ═══ */
.particles{position:fixed;top:0;left:0;width:100%%;height:100%%;pointer-events:none;z-index:0;overflow:hidden}
.particles::before,.particles::after{content:'';position:absolute;width:400px;height:400px;border-radius:50%%;filter:blur(120px);animation:float 12s ease-in-out infinite}
.particles::before{background:rgba(88,166,255,0.06);top:10%%;left:10%%;animation-delay:0s}
.particles::after{background:rgba(124,58,237,0.05);bottom:10%%;right:10%%;animation-delay:6s}
@keyframes float{0%%,100%%{transform:translate(0,0) scale(1)}50%%{transform:translate(30px,-20px) scale(1.1)}}

/* ═══ HEADER ═══ */
.header{
  position:relative;z-index:10;
  display:flex;align-items:center;justify-content:space-between;
  padding:12px 28px;height:60px;
  background:linear-gradient(135deg,rgba(6,10,16,0.95),rgba(15,25,40,0.95));
  border-bottom:1px solid var(--border);
  backdrop-filter:blur(20px);
}
.header-left{display:flex;align-items:center;gap:18px}
.logo{font-size:17px;font-weight:800;letter-spacing:-0.5px}
.logo .brand{background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.logo .sub{color:var(--text-dim);font-weight:400;font-size:11px;margin-left:8px;letter-spacing:0.5px}

.live-badge{
  display:flex;align-items:center;gap:8px;
  font-size:9px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
  padding:6px 16px;border-radius:24px;
  background:rgba("""
        + ("239,68,68" if has_anomalies else "34,197,94")
        + """,0.08);
  border:1px solid rgba("""
        + ("239,68,68" if has_anomalies else "34,197,94")
        + """,0.25);
  color:"""
        + live_color + """;
}
.live-dot{
  width:7px;height:7px;border-radius:50%%;
  background:"""
        + live_color + """;
  animation:pulse 1.8s ease-in-out infinite;
}
@keyframes pulse{0%%,100%%{box-shadow:0 0 0 0 """
        + live_color + """66}50%%{box-shadow:0 0 0 8px """
        + live_color + """00}}

.stats-row{display:flex;align-items:center;gap:8px}
.s-card{
  display:flex;flex-direction:column;align-items:center;
  padding:4px 14px;
  background:var(--bg-glass);
  border:1px solid var(--border);border-radius:10px;
  backdrop-filter:blur(10px);min-width:64px;
}
.s-card .s-label{font-size:7px;color:var(--text-dim);text-transform:uppercase;letter-spacing:1.2px;margin-bottom:1px}
.s-card .s-val{font-size:15px;font-weight:700;font-family:'JetBrains Mono',monospace}
.c-blue{color:var(--accent)}.c-cyan{color:var(--cyan)}.c-red{color:var(--red)}.c-green{color:var(--green)}.c-purple{color:var(--purple)}.c-orange{color:var(--orange)}

/* ═══ LAYOUT ═══ */
.main{position:relative;z-index:5;display:flex;height:calc(100vh - 60px)}
#graph-canvas{flex:1;background:transparent;position:relative}

/* ═══ SIDE PANELS ═══ */
.panel{
  width:340px;background:var(--bg-card);
  border-left:1px solid var(--border);
  overflow-y:auto;display:flex;flex-direction:column;
  backdrop-filter:blur(20px);
}
.panel-section{padding:14px 16px;border-bottom:1px solid var(--border)}
.panel-heading{
  font-size:9px;font-weight:700;
  text-transform:uppercase;letter-spacing:2px;
  margin-bottom:10px;display:flex;align-items:center;gap:8px;
}
.panel-heading .dot{width:6px;height:6px;border-radius:50%%}

/* ═══ GLASS CARDS ═══ */
.glass-card{
  background:var(--bg-glass);
  border:1px solid var(--border);
  border-radius:12px;padding:14px;
  backdrop-filter:blur(12px);
  transition:all .2s ease;margin-bottom:8px;
}
.glass-card:hover{border-color:var(--border-glow);box-shadow:0 0 20px rgba(88,166,255,0.08)}
.glass-card.threat{border-color:rgba(239,68,68,0.3)}
.glass-card.threat:hover{box-shadow:0 0 20px rgba(239,68,68,0.1)}

/* ═══ ATTACK TIMELINE ═══ */
.atk-item{
  display:flex;gap:12px;padding:10px 12px;
  border-radius:10px;cursor:pointer;
  background:rgba(239,68,68,0.04);
  border:1px solid rgba(239,68,68,0.1);
  margin-bottom:6px;transition:all .15s;
}
.atk-item:hover{background:rgba(239,68,68,0.1);border-color:rgba(239,68,68,0.25)}
.atk-icon{
  width:32px;height:32px;border-radius:8px;
  display:flex;align-items:center;justify-content:center;
  font-size:14px;flex-shrink:0;
  background:rgba(239,68,68,0.1);
}
.atk-info{flex:1;min-width:0}
.atk-name{font-size:11px;font-weight:600;color:var(--text-bright);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.atk-meta{font-size:9px;color:var(--text-dim);margin-top:2px}
.mitre-tag{
  font-size:8px;font-weight:600;padding:2px 7px;
  border-radius:4px;letter-spacing:.3px;
  background:rgba(124,58,237,0.15);color:var(--purple);
  display:inline-block;margin-top:3px;
}
.sev-tag{
  font-size:8px;font-weight:700;padding:2px 7px;
  border-radius:4px;letter-spacing:.5px;
}
.sev-tag.critical{background:rgba(239,68,68,0.15);color:var(--red)}
.sev-tag.high{background:rgba(245,158,11,0.15);color:var(--orange)}
.sev-tag.medium{background:rgba(6,182,212,0.15);color:var(--cyan)}

/* ═══ DUAL-PLANE BARS ═══ */
.dp-bar-wrap{margin-top:8px}
.dp-bar{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.dp-bar .lbl{font-size:9px;width:85px;color:var(--text-dim);text-align:right}
.dp-bar .track{flex:1;height:8px;background:rgba(255,255,255,0.04);border-radius:4px;overflow:hidden}
.dp-bar .fill{height:100%%;border-radius:4px;transition:width .6s ease}
.dp-bar .cnt{font-size:10px;font-weight:600;width:28px;font-family:'JetBrains Mono',monospace}

/* ═══ CONFIDENCE GAUGE ═══ */
.gauge-wrap{display:flex;align-items:center;justify-content:center;padding:8px 0}
.gauge{position:relative;width:100px;height:50px;overflow:hidden}
.gauge-bg{
  width:100px;height:100px;border-radius:50%%;
  border:6px solid rgba(255,255,255,0.05);
  border-bottom-color:transparent;border-left-color:transparent;
  transform:rotate(-90deg);
  position:absolute;top:0;left:0;
}
.gauge-fill{
  width:100px;height:100px;border-radius:50%%;
  border:6px solid transparent;
  border-top-color:var(--accent);border-right-color:var(--accent);
  position:absolute;top:0;left:0;
  transition:transform .8s cubic-bezier(0.4,0,0.2,1);
}
.gauge-val{
  position:absolute;bottom:0;left:50%%;transform:translateX(-50%%);
  font-size:18px;font-weight:800;font-family:'JetBrains Mono',monospace;
  color:var(--text-bright);
}
.gauge-label{font-size:8px;color:var(--text-dim);text-align:center;text-transform:uppercase;letter-spacing:1px;margin-top:4px}

/* ═══ ALERT ITEMS ═══ */
.alert-item{
  display:flex;align-items:center;gap:10px;
  padding:8px 10px;border-radius:8px;cursor:pointer;
  background:rgba(239,68,68,0.04);
  border:1px solid rgba(239,68,68,0.1);
  margin-bottom:5px;transition:all .15s;
}
.alert-item:hover{background:rgba(239,68,68,0.1);border-color:rgba(239,68,68,0.25)}
.a-dot{
  width:7px;height:7px;border-radius:50%%;background:var(--red);flex-shrink:0;
  animation:aDot 2s infinite;
}
@keyframes aDot{0%%,100%%{box-shadow:0 0 0 0 rgba(239,68,68,0.5)}50%%{box-shadow:0 0 0 5px rgba(239,68,68,0)}}
.a-info{flex:1}
.a-name{font-size:11px;font-weight:600;color:var(--text-bright)}
.a-score{font-size:9px;color:var(--text-dim);font-family:'JetBrains Mono',monospace}
.a-sev{font-size:8px;font-weight:700;padding:2px 7px;border-radius:4px;letter-spacing:.5px}
.a-sev.HIGH{background:rgba(239,68,68,0.15);color:var(--red)}
.a-sev.MEDIUM{background:rgba(245,158,11,0.15);color:var(--orange)}

/* ═══ NODE INSPECTOR ═══ */
.nd-card{
  background:var(--bg-glass);border:1px solid var(--border);
  border-radius:12px;padding:14px;backdrop-filter:blur(12px);
}
.nd-card.anom{border-color:rgba(239,68,68,0.3)}
.nd-head{display:flex;align-items:center;gap:8px;margin-bottom:10px}
.nd-id{font-size:18px;font-weight:800;font-family:'JetBrains Mono',monospace}
.nd-badge{
  font-size:8px;font-weight:700;padding:3px 10px;
  border-radius:14px;text-transform:uppercase;letter-spacing:.5px;
}
.nd-badge.anom{background:rgba(239,68,68,0.12);color:var(--red)}
.nd-badge.norm{background:rgba(34,197,94,0.12);color:var(--green)}
.nd-badge.feat{background:rgba(245,158,11,0.12);color:var(--orange)}
.nd-badge.struct{background:rgba(167,139,250,0.12);color:var(--purple)}
.nd-badge.dual{background:rgba(239,68,68,0.15);color:var(--red)}
.d-row{
  display:flex;justify-content:space-between;
  padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.04);font-size:11px;
}
.d-row .dk{color:var(--text-dim)}
.d-row .dv{font-weight:500;font-family:'JetBrains Mono',monospace;font-size:11px}

/* Feature bars */
.fb-wrap{margin-top:8px}
.fb-r{display:flex;align-items:center;gap:5px;margin-bottom:3px;font-size:9px}
.fb-r .fl{width:95px;color:var(--text-dim);text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.fb-r .bg{flex:1;height:5px;background:rgba(255,255,255,0.04);border-radius:3px;overflow:hidden}
.fb-r .fi{height:100%%;border-radius:3px;transition:width .4s ease}
.fb-r .fi.norm{background:linear-gradient(90deg,#0a5c2a,var(--green))}
.fb-r .fi.warn{background:linear-gradient(90deg,#7a4a00,var(--orange))}
.fb-r .fi.anom{background:linear-gradient(90deg,#7a1a1a,var(--red))}
.fb-r .fv{width:38px;text-align:right;color:var(--text-dim);font-family:'JetBrains Mono',monospace}

.placeholder{color:var(--text-dim);font-size:11px;text-align:center;padding:24px 12px;line-height:1.8}

/* ═══ ROOT CAUSE PANEL ═══ */
.rc-text{
  font-size:10px;color:var(--text);line-height:1.6;
  max-height:200px;overflow-y:auto;white-space:pre-wrap;word-wrap:break-word;
  padding:8px;background:rgba(0,0,0,0.2);border-radius:8px;
  font-family:'JetBrains Mono',monospace;
}

.panel::-webkit-scrollbar{width:4px}
.panel::-webkit-scrollbar-thumb{background:rgba(88,166,255,0.2);border-radius:2px}
.panel::-webkit-scrollbar-thumb:hover{background:rgba(88,166,255,0.4)}

div.vis-tooltip{
  font-family:'Inter',sans-serif !important;
  background:rgba(12,18,30,0.95) !important;
  color:var(--text) !important;
  border:1px solid var(--border) !important;
  border-radius:10px !important;
  padding:12px 16px !important;
  box-shadow:0 12px 40px rgba(0,0,0,0.5) !important;
  font-size:12px !important;line-height:1.6 !important;
  backdrop-filter:blur(20px) !important;
}
</style>
</head>
<body>
<div class="particles"></div>

<!-- ═══ HEADER ═══ -->
<div class="header">
  <div class="header-left">
    <div class="logo">
      <span class="brand">AutoNet-GNN</span>
      <span class="sub">Dual Poisoning Defense System</span>
    </div>
    <div class="live-badge"><div class="live-dot"></div>"""
        + live_text + """</div>
  </div>
  <div class="stats-row">
    <div class="s-card"><span class="s-label">Nodes</span><span class="s-val c-blue">"""
        + str(num_nodes) + """</span></div>
    <div class="s-card"><span class="s-label">Edges</span><span class="s-val c-cyan">"""
        + str(num_edges) + """</span></div>
    <div class="s-card"><span class="s-label">Flagged</span><span class="s-val c-red">"""
        + str(num_flagged) + """</span></div>
    <div class="s-card"><span class="s-label">Clean</span><span class="s-val c-green">"""
        + str(num_nodes - num_flagged) + """</span></div>
    <div class="s-card"><span class="s-label">σ Threshold</span><span class="s-val c-purple">"""
        + ("%.3f" % threshold) + """</span></div>
    <div class="s-card"><span class="s-label">Dual</span><span class="s-val c-red">"""
        + str(n_dual) + """</span></div>
    <div class="s-card"><span class="s-label">Model</span><span class="s-val c-cyan" style="font-size:10px">GraphSAGE</span></div>
  </div>
</div>

<!-- ═══ MAIN ═══ -->
<div class="main">
  <div id="graph-canvas"></div>
  <div class="panel">

    <!-- ATTACK TIMELINE -->
    <div class="panel-section">
      <div class="panel-heading"><div class="dot" style="background:var(--red)"></div><span style="color:var(--red)">Attack Timeline</span></div>
      <div id="attack-timeline"></div>
    </div>

    <!-- DUAL-PLANE VERIFICATION -->
    <div class="panel-section">
      <div class="panel-heading"><div class="dot" style="background:var(--accent)"></div><span style="color:var(--accent)">Dual-Plane Verification</span></div>
      <div class="dp-bar-wrap">
        <div class="dp-bar"><span class="lbl">Clean</span><div class="track"><div class="fill" style="width:""" + str(int(n_clean / max(num_nodes, 1) * 100)) + """%;background:var(--green)"></div></div><span class="cnt c-green">""" + str(n_clean) + """</span></div>
        <div class="dp-bar"><span class="lbl">Feature</span><div class="track"><div class="fill" style="width:""" + str(int(n_feat / max(num_nodes, 1) * 100)) + """%;background:var(--orange)"></div></div><span class="cnt c-orange">""" + str(n_feat) + """</span></div>
        <div class="dp-bar"><span class="lbl">Structure</span><div class="track"><div class="fill" style="width:""" + str(int(n_struct / max(num_nodes, 1) * 100)) + """%;background:var(--purple)"></div></div><span class="cnt c-purple">""" + str(n_struct) + """</span></div>
        <div class="dp-bar"><span class="lbl">Dual</span><div class="track"><div class="fill" style="width:""" + str(int(n_dual / max(num_nodes, 1) * 100)) + """%;background:var(--red)"></div></div><span class="cnt c-red">""" + str(n_dual) + """</span></div>
      </div>
      <!-- Confidence Gauge -->
      <div class="gauge-wrap">
        <div>
          <div class="gauge">
            <div class="gauge-bg"></div>
            <div class="gauge-fill" style="transform:rotate(""" + str(-90 + int(conf_pct * 1.8)) + """deg)"></div>
            <div class="gauge-val">""" + str(conf_pct) + """%%</div>
          </div>
          <div class="gauge-label">Neural Confidence</div>
        </div>
      </div>
    </div>

    <!-- THREAT ALERTS -->
    <div class="panel-section">
      <div class="panel-heading"><div class="dot" style="background:var(--orange)"></div><span style="color:var(--orange)">Threat Alerts</span></div>
      <div id="alerts-list"></div>
    </div>

    <!-- NODE INSPECTOR -->
    <div class="panel-section" style="flex:1">
      <div class="panel-heading"><div class="dot" style="background:var(--cyan)"></div><span style="color:var(--cyan)">Node Inspector</span></div>
      <div id="node-detail">
        <div class="placeholder">Click any node in the graph<br>to inspect its telemetry and<br>poisoning classification</div>
      </div>
    </div>

    <!-- ROOT CAUSE -->
    <div class="panel-section">
      <div class="panel-heading"><div class="dot" style="background:var(--accent2)"></div><span style="color:var(--accent2)">Root Cause Analysis</span></div>
      <div class="rc-text" id="root-cause-text">Loading analysis...</div>
    </div>

  </div>
</div>

<script>
var VIS_NODES = """ + vis_nodes_json + """;
var VIS_EDGES = """ + vis_edges_json + """;
var NODE_META = """ + node_meta_json + """;
var ALERTS = """ + alerts_json + """;
var ATTACKS = """ + attacks_json + """;
var THRESHOLD = """ + str(threshold) + """;
var ROOT_CAUSE = """ + rc_escaped + """;

/* ═══ ROOT CAUSE TEXT ═══ */
document.getElementById('root-cause-text').textContent = ROOT_CAUSE || 'Analysis pending...';

/* ═══ GRAPH ═══ */
var container = document.getElementById('graph-canvas');
var network = new vis.Network(container, {
  nodes: new vis.DataSet(VIS_NODES),
  edges: new vis.DataSet(VIS_EDGES)
}, {
  physics: {
    barnesHut: {
      gravitationalConstant: -20000,
      centralGravity: 0.15,
      springLength: 200,
      springConstant: 0.04,
      damping: 0.12
    },
    solver: 'barnesHut',
    stabilization: { iterations: 300, fit: true }
  },
  interaction: { hover: true, tooltipDelay: 80, zoomView: true, dragView: true, dragNodes: true },
  nodes: { shape: 'dot', font: { face: 'JetBrains Mono, monospace' } },
  edges: { smooth: { type: 'continuous', roundness: 0.12 } }
});

/* ── Tooltips ── */
network.on('hoverNode', function(p) {
  var n = NODE_META[p.node]; if (!n) return;
  var status = n.is_anomaly ? 'ANOMALY' : 'NORMAL';
  var sColor = n.is_anomaly ? '#ef4444' : '#22c55e';
  var cls = n.classification || 'UNKNOWN';
  var tip = '<div style="min-width:180px">'
    + '<div style="font-weight:700;font-size:13px;margin-bottom:4px">Node ' + n.id + '</div>'
    + '<div>Score: <span style="color:' + sColor + ';font-weight:600">' + n.score.toFixed(4) + '</span></div>'
    + '<div>Class: <span style="font-weight:600">' + cls + '</span></div>'
    + '<div>Neighbors: <span style="font-weight:500">' + n.neighbors.length + '</span></div>'
    + '</div>';
  network.body.data.nodes.update({ id: p.node, title: tip });
});

/* ═══ ATTACK TIMELINE ═══ */
(function() {
  var el = document.getElementById('attack-timeline');
  if (!ATTACKS.length) { el.innerHTML = '<div class="placeholder">No attacks simulated</div>'; return; }
  var icons = {'Denial of Service':'🌊','Man-in-the-Middle':'🔗','Reconnaissance':'🔍',
    'Credential Access':'🔑','Initial Access':'💉','Lateral Movement':'🕸','Exfiltration':'📤','Impact':'💥'};
  var h = '';
  for (var i = 0; i < ATTACKS.length; i++) {
    var a = ATTACKS[i];
    var cat = a.category || '';
    var icon = '⚡';
    for (var k in icons) { if (cat.indexOf(k) >= 0) { icon = icons[k]; break; } }
    h += '<div class="atk-item">'
      + '<div class="atk-icon">' + icon + '</div>'
      + '<div class="atk-info"><div class="atk-name">' + a.attack_name + '</div>'
      + '<div class="atk-meta">' + a.attack_id + ' · ' + cat + '</div>'
      + '<span class="mitre-tag">' + (a.mitre_att_ck || 'N/A') + '</span> '
      + '<span class="sev-tag ' + (a.severity || '') + '">' + (a.severity || '?').toUpperCase() + '</span>'
      + '</div></div>';
  }
  el.innerHTML = h;
})();

/* ═══ ALERTS ═══ */
(function() {
  var el = document.getElementById('alerts-list');
  if (!ALERTS.length) { el.innerHTML = '<div class="placeholder">No active threats</div>'; return; }
  var h = '';
  for (var i = 0; i < ALERTS.length; i++) {
    var a = ALERTS[i];
    h += '<div class="alert-item" onclick="inspectNode(' + a.node_id + ');focusNode(' + a.node_id + ')">'
      + '<div class="a-dot"></div>'
      + '<div class="a-info"><div class="a-name">Node ' + a.node_id + ' &mdash; ' + a.alert_type.replace('_',' ') + '</div>'
      + '<div class="a-score">score: ' + a.anomaly_score.toFixed(4) + '</div></div>'
      + '<div class="a-sev ' + a.severity + '">' + a.severity + '</div></div>';
  }
  el.innerHTML = h;
})();

/* ═══ FOCUS ═══ */
function focusNode(id) {
  network.selectNodes([id]);
  network.focus(id, { scale: 1.5, animation: { duration: 600, easingFunction: 'easeInOutQuad' } });
}

/* ═══ CLICK ═══ */
network.on('click', function(p) { if (p.nodes.length) inspectNode(p.nodes[0]); });

/* ═══ NODE INSPECTOR ═══ */
function inspectNode(nid) {
  var n = NODE_META[nid]; if (!n) return;
  var isA = n.is_anomaly;
  var cls = n.classification || 'UNKNOWN';
  var clsMap = {'CLEAN':'norm','FEATURE_POISONED':'feat','STRUCTURE_POISONED':'struct','DUAL_POISONED':'dual'};
  var bc = 'nd-badge ' + (clsMap[cls] || (isA ? 'anom' : 'norm'));
  var cc = isA ? 'nd-card anom' : 'nd-card';
  var sc = isA ? '#ef4444' : '#22c55e';

  var fkeys = Object.keys(n.features);
  var bars = '';
  for (var i = 0; i < fkeys.length; i++) {
    var fn = fkeys[i], fv = n.features[fn];
    var pct = Math.min(fv * 100, 100);
    var fc = fv > 0.6 ? 'fi anom' : (fv > 0.3 ? 'fi warn' : 'fi norm');
    bars += '<div class="fb-r"><span class="fl">' + fn + '</span>'
      + '<div class="bg"><div class="' + fc + '" style="width:' + pct + '%"></div></div>'
      + '<span class="fv">' + fv.toFixed(3) + '</span></div>';
  }

  var nbrs = n.neighbors.length > 12 ? n.neighbors.slice(0,12).join(', ') + ' ...' : n.neighbors.join(', ');
  var conf = n.confidence ? (n.confidence * 100).toFixed(0) + '%' : 'N/A';

  document.getElementById('node-detail').innerHTML =
    '<div class="' + cc + '">'
    + '<div class="nd-head"><span class="nd-id">Node ' + n.id + '</span><span class="' + bc + '">' + cls + '</span></div>'
    + '<div class="d-row"><span class="dk">Device</span><span class="dv">' + n.device_type + '</span></div>'
    + '<div class="d-row"><span class="dk">Score</span><span class="dv" style="color:' + sc + '">' + n.score.toFixed(4) + '</span></div>'
    + '<div class="d-row"><span class="dk">Threshold</span><span class="dv">' + THRESHOLD.toFixed(4) + '</span></div>'
    + '<div class="d-row"><span class="dk">Confidence</span><span class="dv">' + conf + '</span></div>'
    + '<div class="d-row"><span class="dk">Severity</span><span class="dv">' + (n.severity||'NORMAL') + '</span></div>'
    + '<div class="d-row"><span class="dk">Neighbors</span><span class="dv">' + n.neighbors.length + '</span></div>'
    + '<div class="d-row"><span class="dk">IDs</span><span class="dv" style="font-size:9px">' + nbrs + '</span></div>'
    + '</div>'
    + '<div style="margin-top:10px"><div class="panel-heading" style="color:var(--cyan)"><div class="dot" style="background:var(--cyan)"></div>Feature Vector</div>'
    + '<div class="fb-wrap">' + bars + '</div></div>';
}
</script>
</body>
</html>"""
    )
