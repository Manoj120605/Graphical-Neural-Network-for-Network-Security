"""
orchestrator.py
===============
Central pipeline coordinator for AutoNet-GNN.

Runs 7 stages:
  1. DISCOVER  — Docker container discovery
  2. SIMULATE  — Interactive or automatic attack selection
  3. INGEST    — Build graph with enriched telemetry
  4. DETECT    — GNN anomaly detection (GraphSAGE + L2 deviation)
  5. VERIFY    — Dual-plane structural poisoning defense
  6. ANALYZE   — Automated RAG root-cause analysis
  7. REPORT    — Generate main log + dashboard
"""

import os
import sys
import time
import json
from datetime import datetime, timezone

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

OUTPUT_DIR = os.path.join(_ROOT, "syntheticdata")
ATTACK_DIR = os.path.join(_ROOT, "synthetic-attacks")


def _banner(msg: str, char="═"):
    width = 64
    print("\n" + char * width)
    print("  " + msg)
    print(char * width)


def _stage_header(stage_num: int, total: int, title: str, desc: str):
    print("\n┌─ Stage %d/%d: %s ─────────────────────────────────" % (stage_num, total, title))
    print("│  %s" % desc)
    print("└" + "─" * 56)


# ═══════════════════════════════════════════════════════════════════════
# STAGE 1: DISCOVER
# ═══════════════════════════════════════════════════════════════════════

def stage_discover():
    """Discover Docker containers (spine/leaf nodes)."""
    from Node_Creation.autonet_core import discover_nodes
    nodes = discover_nodes()
    if not nodes:
        raise RuntimeError(
            "No spine/leaf containers found. "
            "Is Docker running and is the docker-compose stack up?"
        )
    return nodes


# ═══════════════════════════════════════════════════════════════════════
# STAGE 2: SIMULATE (interactive attack selection)
# ═══════════════════════════════════════════════════════════════════════

def stage_simulate_interactive(nodes):
    """Show attack menu, let user pick, then simulate."""
    from Node_Creation.attack_simulator import (
        load_attack_patterns, select_attack_scenarios,
        generate_attack_telemetry, save_attack_log,
    )

    patterns = load_attack_patterns(ATTACK_DIR)
    if not patterns:
        raise RuntimeError("No attack patterns found in %s" % ATTACK_DIR)

    # Interactive menu
    print("\n┌─ Available Attack Patterns ───────────────────────┐")
    for i, p in enumerate(patterns, 1):
        sev_color = {"critical": "🔴", "high": "🟠", "medium": "🟡"}.get(p.get("severity", ""), "⚪")
        print("│  %2d. [%s] %s %s" % (
            i, p.get("id", "?"), sev_color, p.get("name", "Unknown")))
        print("│      %s | %s" % (p.get("category", ""), p.get("mitre_att_ck", "")))
    print("│")
    print("│  Enter numbers (e.g. 1,3,5) or 'all' or 'random'")
    print("└───────────────────────────────────────────────────┘")

    choice = input("\n  Select attacks > ").strip().lower()

    if choice == "all":
        selected = patterns
    elif choice == "random":
        selected = select_attack_scenarios(patterns, count=3)
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(",")]
            selected = [patterns[i] for i in indices if 0 <= i < len(patterns)]
        except (ValueError, IndexError):
            print("  Invalid selection, using random 3 attacks.")
            selected = select_attack_scenarios(patterns, count=3)

    if not selected:
        selected = select_attack_scenarios(patterns, count=3)

    print("\n  Selected %d attacks: %s" % (
        len(selected), ", ".join(p.get("id", "?") for p in selected)))

    node_names = [n["name"] for n in nodes]
    telemetry, attack_log = generate_attack_telemetry(
        node_names, selected, victims_per_attack=2,
    )
    save_attack_log(attack_log, OUTPUT_DIR)

    return telemetry, attack_log


def stage_simulate_auto(nodes, attack_ids=None, num_attacks=3):
    """Automatic attack simulation (no prompts)."""
    from Node_Creation.attack_simulator import (
        load_attack_patterns, select_attack_scenarios,
        generate_attack_telemetry, save_attack_log,
    )

    patterns = load_attack_patterns(ATTACK_DIR)
    if not patterns:
        raise RuntimeError("No attack patterns found in %s" % ATTACK_DIR)

    if attack_ids:
        selected = [p for p in patterns if p.get("id") in attack_ids]
        if not selected:
            print("  WARNING: specified attack IDs not found, using random")
            selected = select_attack_scenarios(patterns, count=num_attacks)
    else:
        selected = select_attack_scenarios(patterns, count=num_attacks)

    node_names = [n["name"] for n in nodes]
    telemetry, attack_log = generate_attack_telemetry(
        node_names, selected, victims_per_attack=2,
    )
    save_attack_log(attack_log, OUTPUT_DIR)

    return telemetry, attack_log


# ═══════════════════════════════════════════════════════════════════════
# STAGE 3: INGEST
# ═══════════════════════════════════════════════════════════════════════

def stage_ingest(attack_telemetry=None, attack_log=None):
    """Build graph from telemetry."""
    from data.ingest_docker import ingest
    return ingest(
        output_dir=OUTPUT_DIR,
        attack_telemetry=attack_telemetry,
        attack_log=attack_log,
    )


# ═══════════════════════════════════════════════════════════════════════
# STAGE 4: DETECT
# ═══════════════════════════════════════════════════════════════════════

def stage_detect(data):
    """Run GNN anomaly detection."""
    from detection import detect
    return detect(data, output_dir=OUTPUT_DIR)


# ═══════════════════════════════════════════════════════════════════════
# STAGE 5: VERIFY (Dual-Plane)
# ═══════════════════════════════════════════════════════════════════════

def stage_verify(data, scores, threshold):
    """Dual-plane structural poisoning verification."""
    from detection.dual_plane_verify import verify
    return verify(data, scores, threshold, output_dir=OUTPUT_DIR)


# ═══════════════════════════════════════════════════════════════════════
# STAGE 6: ANALYZE (Automated RAG)
# ═══════════════════════════════════════════════════════════════════════

def stage_analyze():
    """Automated RAG root-cause analysis."""
    from rag.auto_analyze import auto_analyze
    return auto_analyze(rebuild_kb=True)


# ═══════════════════════════════════════════════════════════════════════
# STAGE 7: REPORT + DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

def stage_report(pipeline_stats, data, scores, flagged, threshold,
                 classifications):
    """Generate report and dashboard."""
    from reporting.generate_report import generate_report
    from visualization import visualize_interactive

    # Dashboard
    viz_dir = os.path.join(_ROOT, "visualization")
    dashboard_path = visualize_interactive(
        data, scores, flagged, threshold,
        output_dir=viz_dir,
        classifications=classifications,
    )

    # Main log
    report_path = generate_report(pipeline_stats, output_dir=OUTPUT_DIR)

    return report_path, dashboard_path


# ═══════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

def run_pipeline(auto_mode: bool = False,
                 attack_ids: list = None,
                 interactive_rag: bool = False,
                 skip_rag: bool = False) -> dict:
    """
    Execute the full AutoNet-GNN pipeline.

    Args:
        auto_mode:       If True, skip interactive attack selection
        attack_ids:      Specific attack IDs to simulate (e.g. ["ATK-001"])
        interactive_rag: If True, also launch interactive RAG after analysis
        skip_rag:        If True, skip RAG analysis entirely

    Returns:
        dict of pipeline results
    """
    total_stages = 7 if not skip_rag else 6
    pipeline_start = time.time()
    start_iso = datetime.now(timezone.utc).isoformat()

    _banner("AutoNet-GNN · Unified Pipeline · Dual Poisoning Defense")

    # ── Stage 1: Discover ──
    _stage_header(1, total_stages, "DISCOVER", "Finding Docker containers (spine/leaf nodes)")
    nodes = stage_discover()

    # ── Stage 2: Simulate ──
    _stage_header(2, total_stages, "SIMULATE", "Attack simulation from synthetic patterns")
    if auto_mode:
        telemetry, attack_log = stage_simulate_auto(nodes, attack_ids=attack_ids)
    else:
        telemetry, attack_log = stage_simulate_interactive(nodes)

    # ── Stage 3: Ingest ──
    _stage_header(3, total_stages, "INGEST", "Building bipartite graph with enriched 16D telemetry")
    data = stage_ingest(attack_telemetry=telemetry, attack_log=attack_log)

    # ── Stage 4: Detect ──
    _stage_header(4, total_stages, "DETECT", "GraphSAGE (16→32→16) + L2 neighborhood deviation")
    scores, flagged, threshold = stage_detect(data)

    # ── Stage 5: Verify ──
    _stage_header(5, total_stages, "VERIFY", "Dual-Plane Structural Poisoning Defense")
    classifications = stage_verify(data, scores, threshold)

    # ── Stage 6: Analyze ──
    rag_report = {"status": "skipped", "analyses": {}}
    if not skip_rag:
        _stage_header(6, total_stages, "ANALYZE", "Automated RAG root-cause analysis (llama3.2)")
        rag_report = stage_analyze()

    # Build pipeline stats
    pipeline_end = time.time()
    end_iso = datetime.now(timezone.utc).isoformat()
    duration = round(pipeline_end - pipeline_start, 2)

    # Load dual-plane report for stats
    dp_report_path = os.path.join(OUTPUT_DIR, "dual_plane_report.json")
    dp_summary = {}
    dp_nodes = {}
    if os.path.exists(dp_report_path):
        with open(dp_report_path, "r") as f:
            dp_data = json.load(f)
        dp_summary = dp_data.get("summary", {})
        dp_nodes = dp_data.get("nodes", {})

    # Build per-node scores
    node_scores = {}
    for nid in flagged:
        node_scores[str(nid)] = round(scores[nid].item(), 6)

    pipeline_stats = {
        "start_time": start_iso,
        "end_time": end_iso,
        "duration_seconds": duration,
        "num_nodes": data.num_nodes,
        "num_edges": data.edge_index.shape[1] // 2,
        "num_flagged": len(flagged),
        "threshold": round(threshold, 6),
        "flagged_nodes": flagged,
        "scores_summary": {
            "mean": round(scores.mean().item(), 6),
            "std": round(scores.std().item(), 6),
            "max": round(scores.max().item(), 6),
            "min": round(scores.min().item(), 6),
            "node_scores": node_scores,
        },
        "attack_log": attack_log,
        "dual_plane_summary": dp_summary,
        "dual_plane_nodes": dp_nodes,
        "rag_report": rag_report,
    }

    # ── Stage 7: Report ──
    stage_num = total_stages
    _stage_header(stage_num, total_stages, "REPORT",
                  "Generating main log and interactive dashboard")
    report_path, dashboard_path = stage_report(
        pipeline_stats, data, scores, flagged, threshold, classifications,
    )

    # ── Summary ──
    _banner("Pipeline Complete · %.1fs" % duration)
    print("  📊 Dashboard  : %s" % dashboard_path)
    print("  📋 Report Log : %s" % report_path)
    print("  🔍 Scores     : %s" % os.path.join(OUTPUT_DIR, "anomaly_scores.csv"))
    print("  🚨 Alerts     : %s" % os.path.join(OUTPUT_DIR, "alerts.json"))
    print("  🎯 Attacks    : %s" % os.path.join(OUTPUT_DIR, "attack_log.json"))
    print("  🛡 Dual-Plane : %s" % os.path.join(OUTPUT_DIR, "dual_plane_report.json"))
    if not skip_rag:
        print("  🤖 Root Cause : %s" % os.path.join(OUTPUT_DIR, "root_cause_report.json"))
    print()

    # ── Optional interactive RAG ──
    if interactive_rag and not skip_rag:
        print("  Launching interactive RAG agent ...")
        try:
            from rag.rag_main import main as rag_main
            rag_main()
        except Exception as e:
            print("  RAG agent failed: %s" % e)

    return pipeline_stats
