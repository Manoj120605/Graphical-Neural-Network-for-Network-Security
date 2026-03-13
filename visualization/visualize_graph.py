"""
visualize_graph.py
==================
Two visualization outputs:
  1. Static PNG  -- visualize()            (matplotlib)
  2. Interactive -- visualize_interactive() (vis.js SOC dashboard)
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
    print("")
    print("[viz] Static PNG : %s" % out_path)
    return out_path


# ──────────────────── INTERACTIVE SOC DASHBOARD ───────────────────────

def visualize_interactive(data: Data,
                          scores: torch.Tensor,
                          flagged: list[int],
                          threshold: float,
                          output_dir: str = "visualization") -> str:
    """
    Generate a self-contained SOC-style interactive HTML dashboard.
    Vis.js loaded from CDN, no server required.
    """
    os.makedirs(output_dir, exist_ok=True)

    num_nodes = data.num_nodes
    num_edges = data.edge_index.shape[1] // 2
    flagged_set = set(flagged)
    max_score = scores.max().item() or 1.0

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

        node_meta[n] = {
            "id": n, "score": round(score, 4),
            "label": int(data.y[n].item()), "is_anomaly": is_flagged,
            "device_type": device, "neighbors": neighbors,
            "features": features,
        }

        if is_flagged:
            size = 28 + 30 * (score / max_score)
            vis_nodes.append({
                "id": n, "label": str(n), "size": size,
                "color": {
                    "background": "#ff4d4d", "border": "#ff1a1a",
                    "highlight": {"background": "#ff6b6b", "border": "#ffffff"},
                    "hover": {"background": "#ff6b6b", "border": "#ffffff"},
                },
                "shadow": {
                    "enabled": True, "color": "rgba(255,77,77,0.8)",
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
                    "background": "#2ecc71", "border": "#27ae60",
                    "highlight": {"background": "#58d68d", "border": "#ffffff"},
                    "hover": {"background": "#58d68d", "border": "#ffffff"},
                },
                "shadow": {"enabled": False},
                "borderWidth": 1.5,
                "font": {"size": 11, "color": "#d0d0d0"},
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
                              "color": {"color": "#ff4d4d", "opacity": 0.55},
                              "width": 2.5})
        else:
            vis_edges.append({"from": s, "to": d,
                              "color": {"color": "#1e3a5f", "opacity": 0.45},
                              "width": 1.0})

    # ── Alerts ──
    alerts_path = os.path.join("syntheticdata", "alerts.json")
    alerts_list = []
    if os.path.exists(alerts_path):
        with open(alerts_path, "r", encoding="utf-8") as f:
            alerts_list = json.load(f).get("alerts", [])

    has_anomalies = len(flagged) > 0

    html = _soc_html(
        vis_nodes_json=json.dumps(vis_nodes),
        vis_edges_json=json.dumps(vis_edges),
        node_meta_json=json.dumps(node_meta),
        alerts_json=json.dumps(alerts_list),
        num_nodes=num_nodes,
        num_edges=num_edges,
        num_flagged=len(flagged),
        threshold=threshold,
        has_anomalies=has_anomalies,
    )

    out_path = os.path.join(output_dir, "dashboard.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print("")
    print("[viz] Dashboard : %s" % out_path)
    return out_path


def _soc_html(vis_nodes_json, vis_edges_json, node_meta_json,
              alerts_json, num_nodes, num_edges, num_flagged,
              threshold, has_anomalies):

    live_color = "#ff4d4d" if has_anomalies else "#2ecc71"
    live_text = "THREAT DETECTED" if has_anomalies else "ALL CLEAR"

    return (
        """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AutoNet-GNN | SOC Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',system-ui,sans-serif;background:#0b0f14;color:#c8ccd4;overflow:hidden;height:100vh}

/* ═══ HEADER ═══ */
.header{
  display:flex;align-items:center;justify-content:space-between;
  padding:10px 24px;
  background:linear-gradient(90deg,#0d1117 0%%,#111922 100%%);
  border-bottom:1px solid #1a2332;
  height:54px;
}
.header-left{display:flex;align-items:center;gap:16px}
.logo{font-size:16px;font-weight:700;letter-spacing:-0.3px}
.logo .brand{color:#58a6ff}
.logo .sub{color:#6e7681;font-weight:400;font-size:12px;margin-left:6px}

.live-badge{
  display:flex;align-items:center;gap:7px;
  font-size:10px;font-weight:600;letter-spacing:1.2px;
  text-transform:uppercase;
  padding:5px 14px;border-radius:20px;
  background:rgba("""
        + ("255,77,77" if has_anomalies else "46,204,113")
        + """,0.1);
  border:1px solid rgba("""
        + ("255,77,77" if has_anomalies else "46,204,113")
        + """,0.3);
  color:"""
        + live_color
        + """;
}
.live-dot{
  width:7px;height:7px;border-radius:50%%;
  background:"""
        + live_color
        + """;
  animation:livePulse 1.5s ease-in-out infinite;
}
@keyframes livePulse{
  0%%,100%%{box-shadow:0 0 0 0 rgba("""
        + ("255,77,77" if has_anomalies else "46,204,113")
        + """,0.6)}
  50%%{box-shadow:0 0 0 6px rgba("""
        + ("255,77,77" if has_anomalies else "46,204,113")
        + """,0)}
}

.stats-row{display:flex;align-items:center;gap:10px}
.s-card{
  display:flex;flex-direction:column;align-items:center;
  padding:3px 14px;
  background:#0d1117;
  border:1px solid #1a2332;border-radius:8px;
  min-width:68px;
}
.s-card .s-label{font-size:8px;color:#6e7681;text-transform:uppercase;letter-spacing:1px}
.s-card .s-val{font-size:16px;font-weight:700;font-family:'JetBrains Mono',monospace}
.s-card .s-val.c-blue{color:#58a6ff}
.s-card .s-val.c-cyan{color:#3fb8af}
.s-card .s-val.c-red{color:#ff4d4d}
.s-card .s-val.c-green{color:#2ecc71}
.s-card .s-val.c-purple{color:#a78bfa}
.s-card .s-val.c-teal{color:#3fb8af;font-size:11px}

/* ═══ LAYOUT ═══ */
.main{display:flex;height:calc(100vh - 54px)}
#graph-canvas{flex:1;background:#0b0f14;position:relative}

/* ═══ SIDE PANEL ═══ */
.panel{
  width:320px;
  background:#0d1117;
  border-left:1px solid #1a2332;
  overflow-y:auto;
  display:flex;flex-direction:column;
}
.panel-section{padding:14px 16px;border-bottom:1px solid #1a2332}
.panel-heading{
  font-size:10px;font-weight:600;color:#58a6ff;
  text-transform:uppercase;letter-spacing:1.5px;
  margin-bottom:8px;
}

/* ── Alert item ── */
.alert-item{
  display:flex;align-items:center;gap:10px;
  padding:8px 10px;border-radius:6px;cursor:pointer;
  background:rgba(255,77,77,0.05);
  border:1px solid rgba(255,77,77,0.12);
  margin-bottom:5px;
  transition:all .15s;
}
.alert-item:hover{background:rgba(255,77,77,0.12);border-color:rgba(255,77,77,0.3)}
.a-dot{
  width:7px;height:7px;border-radius:50%%;background:#ff4d4d;flex-shrink:0;
  animation:aDotPulse 2s infinite;
}
@keyframes aDotPulse{0%%,100%%{box-shadow:0 0 0 0 rgba(255,77,77,0.5)}50%%{box-shadow:0 0 0 6px rgba(255,77,77,0)}}
.a-info{flex:1}
.a-name{font-size:12px;font-weight:600;color:#e0e0e0}
.a-score{font-size:10px;color:#6e7681;font-family:'JetBrains Mono',monospace}
.a-sev{font-size:8px;font-weight:700;padding:2px 7px;border-radius:3px;letter-spacing:.5px}
.a-sev.HIGH{background:rgba(255,77,77,0.15);color:#ff4d4d}
.a-sev.MEDIUM{background:rgba(249,115,22,0.15);color:#f97316}

/* ── Node detail ── */
.nd-card{
  background:#111922;
  border:1px solid #1a2332;
  border-radius:8px;padding:12px;
}
.nd-card.anom{border-color:rgba(255,77,77,0.35)}
.nd-head{display:flex;align-items:center;gap:8px;margin-bottom:10px}
.nd-id{font-size:18px;font-weight:700;font-family:'JetBrains Mono',monospace}
.nd-badge{
  font-size:8px;font-weight:700;padding:3px 9px;
  border-radius:12px;text-transform:uppercase;letter-spacing:.5px;
}
.nd-badge.anom{background:rgba(255,77,77,0.15);color:#ff4d4d}
.nd-badge.norm{background:rgba(46,204,113,0.15);color:#2ecc71}
.d-row{
  display:flex;justify-content:space-between;
  padding:4px 0;border-bottom:1px solid #1a2332;font-size:11px;
}
.d-row .dk{color:#6e7681}
.d-row .dv{font-weight:500;font-family:'JetBrains Mono',monospace;font-size:11px}

/* Feature bars */
.fb-wrap{margin-top:6px}
.fb-r{display:flex;align-items:center;gap:5px;margin-bottom:3px;font-size:9px}
.fb-r .fl{width:95px;color:#6e7681;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.fb-r .bg{flex:1;height:4px;background:#1a2332;border-radius:2px;overflow:hidden}
.fb-r .fi{height:100%%;border-radius:2px;transition:width .4s ease}
.fb-r .fi.norm{background:linear-gradient(90deg,#1a6b3d,#2ecc71)}
.fb-r .fi.anom{background:linear-gradient(90deg,#cc2a2a,#ff4d4d)}
.fb-r .fv{width:38px;text-align:right;color:#6e7681;font-family:'JetBrains Mono',monospace}

.placeholder{color:#3d4450;font-size:11px;text-align:center;padding:24px 12px;line-height:1.7}

.panel::-webkit-scrollbar{width:3px}
.panel::-webkit-scrollbar-thumb{background:#1a2332;border-radius:2px}

/* ═══ VIS.JS TOOLTIP OVERRIDE ═══ */
div.vis-tooltip{
  font-family:'Inter',sans-serif !important;
  background:#111922 !important;
  color:#c8ccd4 !important;
  border:1px solid #1a2332 !important;
  border-radius:8px !important;
  padding:10px 14px !important;
  box-shadow:0 8px 32px rgba(0,0,0,0.5) !important;
  font-size:12px !important;
  line-height:1.6 !important;
}
</style>
</head>
<body>

<!-- ═══ HEADER ═══ -->
<div class="header">
  <div class="header-left">
    <div class="logo">
      <span class="brand">AutoNet-GNN</span>
      <span class="sub">Cognitive Network Monitor</span>
    </div>
    <div class="live-badge">
      <div class="live-dot"></div>
      """
        + live_text
        + """
    </div>
  </div>
  <div class="stats-row">
    <div class="s-card"><span class="s-label">Nodes</span><span class="s-val c-blue">"""
        + str(num_nodes)
        + """</span></div>
    <div class="s-card"><span class="s-label">Edges</span><span class="s-val c-cyan">"""
        + str(num_edges)
        + """</span></div>
    <div class="s-card"><span class="s-label">Flagged</span><span class="s-val c-red">"""
        + str(num_flagged)
        + """</span></div>
    <div class="s-card"><span class="s-label">Normal</span><span class="s-val c-green">"""
        + str(num_nodes - num_flagged)
        + """</span></div>
    <div class="s-card"><span class="s-label">Threshold</span><span class="s-val c-purple">"""
        + ("%.2f" % threshold)
        + """</span></div>
    <div class="s-card"><span class="s-label">Model</span><span class="s-val c-teal">GraphSAGE</span></div>
  </div>
</div>

<!-- ═══ MAIN ═══ -->
<div class="main">
  <div id="graph-canvas"></div>
  <div class="panel">
    <div class="panel-section">
      <div class="panel-heading">Threat Alerts</div>
      <div id="alerts-list"></div>
    </div>
    <div class="panel-section" style="flex:1">
      <div class="panel-heading">Node Inspector</div>
      <div id="node-detail">
        <div class="placeholder">Click any node in the graph<br>to inspect its telemetry</div>
      </div>
    </div>
  </div>
</div>

<script>
var VIS_NODES = """
        + vis_nodes_json + """;
var VIS_EDGES = """
        + vis_edges_json + """;
var NODE_META = """
        + node_meta_json + """;
var ALERTS = """
        + alerts_json + """;
var THRESHOLD = """
        + str(threshold) + """;

/* ═══ GRAPH ═══ */
var container = document.getElementById('graph-canvas');
var network = new vis.Network(container, {
  nodes: new vis.DataSet(VIS_NODES),
  edges: new vis.DataSet(VIS_EDGES)
}, {
  physics: {
    barnesHut: {
      gravitationalConstant: -25000,
      centralGravity: 0.2,
      springLength: 180,
      springConstant: 0.05,
      damping: 0.15
    },
    solver: 'barnesHut',
    stabilization: { iterations: 250, fit: true }
  },
  interaction: {
    hover: true,
    tooltipDelay: 80,
    zoomView: true,
    dragView: true,
    dragNodes: true
  },
  nodes: {
    shape: 'dot',
    font: { face: 'JetBrains Mono, monospace' }
  },
  edges: {
    smooth: { type: 'continuous', roundness: 0.15 }
  }
});

/* ── Tooltips ── */
network.on('hoverNode', function(p) {
  var n = NODE_META[p.node];
  if (!n) return;
  var status = n.is_anomaly ? 'ANOMALY' : 'NORMAL';
  var sColor = n.is_anomaly ? '#ff4d4d' : '#2ecc71';
  var tip = '<div style="min-width:160px">'
    + '<div style="font-weight:700;font-size:13px;margin-bottom:4px">Node ' + n.id + '</div>'
    + '<div>Score: <span style="color:' + sColor + ';font-weight:600">' + n.score.toFixed(4) + '</span></div>'
    + '<div>Neighbors: <span style="font-weight:500">' + n.neighbors.length + '</span></div>'
    + '<div>Status: <span style="color:' + sColor + ';font-weight:700">' + status + '</span></div>'
    + '</div>';
  var ds = network.body.data.nodes;
  ds.update({ id: p.node, title: tip });
});

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

/* ═══ FOCUS NODE ═══ */
function focusNode(id) {
  network.selectNodes([id]);
  network.focus(id, { scale: 1.5, animation: { duration: 600, easingFunction: 'easeInOutQuad' } });
}

/* ═══ CLICK ═══ */
network.on('click', function(p) { if (p.nodes.length) inspectNode(p.nodes[0]); });

/* ═══ NODE INSPECTOR ═══ */
function inspectNode(nid) {
  var n = NODE_META[nid];
  if (!n) return;
  var isA = n.is_anomaly;
  var cc = isA ? 'nd-card anom' : 'nd-card';
  var bc = isA ? 'nd-badge anom' : 'nd-badge norm';
  var bt = isA ? 'ANOMALY' : 'NORMAL';
  var sc = isA ? '#ff4d4d' : '#2ecc71';

  var fkeys = Object.keys(n.features);
  var mx = 1;
  for (var i = 0; i < fkeys.length; i++) { var av = Math.abs(n.features[fkeys[i]]); if (av > mx) mx = av; }
  var bars = '';
  for (var i = 0; i < fkeys.length; i++) {
    var fn = fkeys[i], fv = n.features[fn];
    var pct = Math.min(Math.abs(fv) / mx * 100, 100);
    var fc = Math.abs(fv) > 5 ? 'fi anom' : 'fi norm';
    bars += '<div class="fb-r"><span class="fl">' + fn + '</span>'
      + '<div class="bg"><div class="' + fc + '" style="width:' + pct + '%%"></div></div>'
      + '<span class="fv">' + fv.toFixed(2) + '</span></div>';
  }

  var nbrs = n.neighbors.length > 15 ? n.neighbors.slice(0,15).join(', ') + ' ...' : n.neighbors.join(', ');

  document.getElementById('node-detail').innerHTML =
    '<div class="' + cc + '">'
    + '<div class="nd-head"><span class="nd-id">Node ' + n.id + '</span><span class="' + bc + '">' + bt + '</span></div>'
    + '<div class="d-row"><span class="dk">Device Type</span><span class="dv">' + n.device_type + '</span></div>'
    + '<div class="d-row"><span class="dk">Anomaly Score</span><span class="dv" style="color:' + sc + '">' + n.score.toFixed(4) + '</span></div>'
    + '<div class="d-row"><span class="dk">Threshold</span><span class="dv">' + THRESHOLD.toFixed(4) + '</span></div>'
    + '<div class="d-row"><span class="dk">Neighbors</span><span class="dv">' + n.neighbors.length + '</span></div>'
    + '<div class="d-row"><span class="dk">Neighbor IDs</span><span class="dv" style="font-size:9px">' + nbrs + '</span></div>'
    + '</div>'
    + '<div style="margin-top:10px"><div class="panel-heading">Feature Vector</div>'
    + '<div class="fb-wrap">' + bars + '</div></div>';
}
</script>
</body>
</html>"""
    )
