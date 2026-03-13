"""
generate_report.py
==================
Produces the main pipeline log:  syntheticdata/autonet_report.log

Consolidates:
  - Pipeline execution summary (timestamps, durations)
  - Attack simulation details (which attacks, victims)
  - GNN detection results (scores, flags, threshold)
  - Dual-plane verification results
  - Root cause analysis from RAG
  - Remediation recommendations
"""

import os
import json
from datetime import datetime, timezone

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "syntheticdata")


def generate_report(pipeline_stats: dict,
                    output_dir: str = None) -> str:
    """
    Generate the main autonet_report.log.

    Args:
        pipeline_stats: dict containing:
            - start_time, end_time
            - num_nodes, num_edges, num_flagged
            - threshold, scores_summary
            - attack_log (list of attack events)
            - dual_plane_summary (classification counts)
            - rag_report (root cause analysis text)
        output_dir: where to write the log

    Returns:
        Path to the generated log file
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "autonet_report.log")
    now = datetime.now(timezone.utc)

    lines = []
    lines.append("=" * 72)
    lines.append("  AutoNet-GNN  ·  Pipeline Execution Report")
    lines.append("  Generated: %s" % now.strftime("%Y-%m-%d %H:%M:%S UTC"))
    lines.append("=" * 72)
    lines.append("")

    # ── Section 1: Pipeline Summary ──
    lines.append("┌─ PIPELINE SUMMARY ──────────────────────────────────────┐")
    lines.append("│")
    start = pipeline_stats.get("start_time", "N/A")
    end = pipeline_stats.get("end_time", "N/A")
    duration = pipeline_stats.get("duration_seconds", "N/A")
    lines.append("│  Start Time   : %s" % start)
    lines.append("│  End Time     : %s" % end)
    lines.append("│  Duration     : %s seconds" % duration)
    lines.append("│  Nodes        : %s" % pipeline_stats.get("num_nodes", "N/A"))
    lines.append("│  Edges        : %s" % pipeline_stats.get("num_edges", "N/A"))
    lines.append("│  GNN Model    : GraphSAGE (16→32→16)")
    lines.append("│  Threshold    : %s (2-sigma)" % pipeline_stats.get("threshold", "N/A"))
    lines.append("│  Flagged      : %s nodes" % pipeline_stats.get("num_flagged", "N/A"))
    lines.append("│")
    lines.append("└──────────────────────────────────────────────────────────┘")
    lines.append("")

    # ── Section 2: Attack Simulation ──
    attack_log = pipeline_stats.get("attack_log", [])
    lines.append("┌─ ATTACK SIMULATION ─────────────────────────────────────┐")
    lines.append("│")
    if attack_log:
        lines.append("│  Attacks Simulated: %d" % len(attack_log))
        lines.append("│")
        for event in attack_log:
            lines.append("│  ▸ [%s] %s" % (event.get("attack_id", "?"), event.get("attack_name", "Unknown")))
            lines.append("│    Category : %s" % event.get("category", "?"))
            lines.append("│    Severity : %s" % event.get("severity", "?"))
            lines.append("│    MITRE    : %s" % event.get("mitre_att_ck", "N/A"))
            lines.append("│    Victims  : %s" % ", ".join(event.get("victim_nodes", [])))
            iocs = event.get("indicators", [])
            if iocs:
                lines.append("│    IoCs     :")
                for ioc in iocs[:4]:
                    lines.append("│      - %s" % ioc)
            lines.append("│")
    else:
        lines.append("│  No attacks simulated.")
        lines.append("│")
    lines.append("└──────────────────────────────────────────────────────────┘")
    lines.append("")

    # ── Section 3: GNN Detection Results ──
    lines.append("┌─ GNN ANOMALY DETECTION ──────────────────────────────────┐")
    lines.append("│")
    scores_summary = pipeline_stats.get("scores_summary", {})
    lines.append("│  Mean Score   : %s" % scores_summary.get("mean", "N/A"))
    lines.append("│  Std Score    : %s" % scores_summary.get("std", "N/A"))
    lines.append("│  Max Score    : %s" % scores_summary.get("max", "N/A"))
    lines.append("│  Min Score    : %s" % scores_summary.get("min", "N/A"))
    lines.append("│  Threshold    : %s" % pipeline_stats.get("threshold", "N/A"))
    lines.append("│")
    flagged_nodes = pipeline_stats.get("flagged_nodes", [])
    if flagged_nodes:
        lines.append("│  Flagged Nodes:")
        for nid in flagged_nodes[:15]:
            score = scores_summary.get("node_scores", {}).get(str(nid), "?")
            lines.append("│    Node %-4s : score = %s" % (nid, score))
    lines.append("│")
    lines.append("└──────────────────────────────────────────────────────────┘")
    lines.append("")

    # ── Section 4: Dual-Plane Verification ──
    dp = pipeline_stats.get("dual_plane_summary", {})
    lines.append("┌─ DUAL-PLANE STRUCTURAL POISONING VERIFICATION ──────────┐")
    lines.append("│")
    lines.append("│  Feature Plane Anomalies  : %s" % dp.get("feature_anomalies", "N/A"))
    lines.append("│  Structure Plane Anomalies: %s" % dp.get("structure_anomalies", "N/A"))
    lines.append("│")
    classifications = dp.get("classifications", {})
    for cls in ["CLEAN", "FEATURE_POISONED", "STRUCTURE_POISONED", "DUAL_POISONED"]:
        if cls in classifications:
            lines.append("│  %-24s: %d nodes" % (cls, classifications[cls]))
    lines.append("│")

    # Per-node details for non-CLEAN nodes
    node_details = pipeline_stats.get("dual_plane_nodes", {})
    non_clean = {k: v for k, v in node_details.items()
                 if v.get("classification") != "CLEAN"}
    if non_clean:
        lines.append("│  Affected Nodes:")
        for nid, info in sorted(non_clean.items(), key=lambda x: x[0]):
            lines.append("│    Node %-4s : %s (confidence: %.0f%%)" % (
                nid, info["classification"], info.get("confidence", 0) * 100))
    lines.append("│")
    lines.append("└──────────────────────────────────────────────────────────┘")
    lines.append("")

    # ── Section 5: Root Cause Analysis ──
    rag_report = pipeline_stats.get("rag_report", {})
    lines.append("┌─ ROOT CAUSE ANALYSIS ────────────────────────────────────┐")
    lines.append("│")
    for qid in ["anomaly_scan", "root_cause", "remediation"]:
        analysis = rag_report.get("analyses", {}).get(qid, {})
        response = analysis.get("response", "No analysis available")
        status = analysis.get("status", "N/A")
        header = qid.upper().replace("_", " ")
        lines.append("│  ── %s (%s) ──" % (header, status))
        for line in response.strip().split("\n"):
            lines.append("│  %s" % line[:68])
        lines.append("│")
    lines.append("└──────────────────────────────────────────────────────────┘")
    lines.append("")

    # ── Footer ──
    lines.append("=" * 72)
    lines.append("  End of Report  |  AutoNet-GNN v2.0  |  Dual Poisoning Defense")
    lines.append("=" * 72)
    lines.append("")

    content = "\n".join(lines)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("[report] Main log saved: %s" % log_path)
    print("[report] Log size: %d lines, %d bytes" % (len(lines), len(content)))

    return log_path
