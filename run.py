#!/usr/bin/env python
"""
run.py -- AutoNet-GNN  Live Pipeline
======================================
Execution order:
    1. Ingest   - discover Docker containers, poll SSH config telemetry
    1b. Attack  - (optional) simulate attacks from synthetic-attacks/ patterns
    2. Detect   - run GraphSAGE encoder + L2 neighborhood deviation scoring
    3. Visualise - generate SOC dashboard (dashboard.html)
    4. Analyse  - launch the agentic RAG interface for interactive Q&A

Dependencies:
    Core   : torch, torch-geometric, networkx, numpy
    Docker : docker  (pip install docker)
    RAG    : langchain, langgraph, chromadb, sentence-transformers
             (only needed for step 4 — skipped if missing)
"""

import os
import sys
import argparse

_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

OUTPUT_DIR = os.path.join(_ROOT, "syntheticdata")
ATTACK_DIR = os.path.join(_ROOT, "synthetic-attacks")


# ── Helpers ──────────────────────────────────────────────────────────────

def _banner(msg: str):
    print("\n" + "=" * 60)
    print("  " + msg)
    print("=" * 60)


def step_ingest(attack_telemetry=None, attack_log=None) -> "torch_geometric.data.Data":
    """Step 1 — Discover Docker containers and build live graph."""
    from data.ingest_docker import ingest
    return ingest(
        output_dir=OUTPUT_DIR,
        attack_telemetry=attack_telemetry,
        attack_log=attack_log,
    )


def step_attack_simulate(nodes) -> tuple:
    """Step 1b — Simulate attacks using synthetic attack patterns."""
    from Node_Creation.attack_simulator import simulate_attacks, save_attack_log

    node_names = [n['name'] for n in nodes]
    telemetry, attack_log = simulate_attacks(
        node_names, ATTACK_DIR,
        num_attacks=3, victims_per_attack=2,
    )
    save_attack_log(attack_log, OUTPUT_DIR)
    return telemetry, attack_log


def step_detect(data) -> tuple:
    """Step 2 — GraphSAGE embedding + anomaly scoring."""
    from detection import detect
    scores, flagged, threshold = detect(data, output_dir=OUTPUT_DIR)
    return scores, flagged, threshold


def step_visualise(data, scores, flagged, threshold) -> str:
    """Step 3 — Generate interactive SOC dashboard."""
    from visualization import visualize_interactive
    out_path = visualize_interactive(
        data, scores, flagged, threshold,
        output_dir=os.path.join(_ROOT, "visualization"),
    )
    return out_path


def step_rag():
    """Step 4 — Launch the agentic RAG interface."""
    try:
        from rag.rag_main import main as rag_main
        rag_main()
    except ImportError as e:
        print("\n[rag] Skipping RAG agent — missing dependency: %s" % e)
        print("[rag] Install with:")
        print("      pip install langchain langchain-community langchain-ollama "
              "langchain-openai langchain-huggingface langgraph chromadb "
              "sentence-transformers rich")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AutoNet-GNN — Live Network Anomaly Detection Pipeline"
    )
    parser.add_argument(
        "--no-rag", action="store_true",
        help="Skip the interactive RAG agent (steps 1-3 only)"
    )
    parser.add_argument(
        "--rag-only", action="store_true",
        help="Skip ingestion/detection, launch RAG on existing data"
    )
    parser.add_argument(
        "--simulate-attacks", action="store_true",
        help="Inject synthetic attack simulations from synthetic-attacks/ patterns"
    )
    args = parser.parse_args()

    _banner("AutoNet-GNN — Live Network Anomaly Detection")

    if args.rag_only:
        print("\n>> Launching RAG agent on existing data...\n")
        step_rag()
        return

    # ── Step 1: Live Ingestion ──
    total_steps = 5 if args.simulate_attacks else 4
    step_num = 1

    print("\n>> Step %d/%d  Ingesting from Docker containers" % (step_num, total_steps))
    print("   (discovering nodes, polling SSH config via docker exec)\n")

    attack_telemetry = None
    attack_log = None

    if args.simulate_attacks:
        # Discover nodes first for attack simulation
        from Node_Creation.autonet_core import discover_nodes
        nodes = discover_nodes()
        if not nodes:
            print("ERROR: No spine/leaf containers found. Is Docker running?")
            return

        step_num += 1
        print("\n>> Step %d/%d  Simulating attacks from synthetic-attacks/" % (step_num, total_steps))
        print("   (injecting anomalous telemetry for selected attack patterns)\n")
        attack_telemetry, attack_log = step_attack_simulate(nodes)

    step_num += 1 if not args.simulate_attacks else 0
    data = step_ingest(
        attack_telemetry=attack_telemetry,
        attack_log=attack_log,
    )

    # ── Step 2: GNN Detection ──
    step_num += 1
    print("\n>> Step %d/%d  Running GraphSAGE anomaly detection" % (step_num, total_steps))
    print("   (encoder 16→32→16, L2 deviation, 2-sigma threshold)\n")
    scores, flagged, threshold = step_detect(data)

    # ── Step 3: Visualisation ──
    step_num += 1
    print("\n>> Step %d/%d  Generating SOC dashboard" % (step_num, total_steps))
    out_path = step_visualise(data, scores, flagged, threshold)

    # ── Summary ──
    _banner("Pipeline complete")
    print("  Graph     : %s" % os.path.join(OUTPUT_DIR, "synthetic_graph.pt"))
    print("  Scores    : %s" % os.path.join(OUTPUT_DIR, "anomaly_scores.csv"))
    print("  Alerts    : %s" % os.path.join(OUTPUT_DIR, "alerts.json"))
    if attack_log:
        print("  Atk Log   : %s" % os.path.join(OUTPUT_DIR, "attack_log.json"))
    print("  Dashboard : %s" % out_path)
    print()

    if args.no_rag:
        print("[info] --no-rag flag set, skipping RAG agent.")
        return

    # ── Step 4: RAG Agent ──
    print(">> Step %d/%d  Launching Agentic RAG interface" % (total_steps, total_steps))
    print("   (type 'help' for commands, 'exit' to quit)\n")
    step_rag()


if __name__ == "__main__":
    main()
