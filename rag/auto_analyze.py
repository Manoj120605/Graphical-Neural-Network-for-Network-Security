"""
auto_analyze.py
===============
Automated (non-interactive) RAG analysis for AutoNet-GNN.

Programmatically queries the agentic RAG agent to produce:
  1. Anomaly summary
  2. Root cause analysis with dual-poisoning context
  3. Remediation recommendations

Writes results to  syntheticdata/root_cause_report.json
"""

import os
import sys
import json
import traceback
from datetime import datetime, timezone

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "syntheticdata")

# Queries to run automatically against the RAG agent
AUTO_QUERIES = [
    {
        "id": "anomaly_scan",
        "prompt": "Query all anomaly detection results and list every flagged node with its score, severity, and classification.",
    },
    {
        "id": "root_cause",
        "prompt": (
            "Analyze the root cause of the most critical anomalies: "
            "query the anomaly scores, explain the top flagged nodes, "
            "search the knowledge base for matching attack patterns and "
            "dual poisoning indicators, and summarize what is most likely "
            "causing each anomaly. Classify each as feature poisoning, "
            "structure poisoning, or dual poisoning."
        ),
    },
    {
        "id": "remediation",
        "prompt": (
            "Generate a comprehensive remediation action plan for all "
            "detected anomalies. Include immediate containment steps, "
            "investigation actions, and long-term hardening measures. "
            "Prioritize by severity."
        ),
    },
]


def auto_analyze(rebuild_kb: bool = True) -> dict:
    """
    Run automated RAG analysis without any interactive prompts.

    Uses llama3.2 via Ollama by default.

    Returns:
        dict with keys: anomaly_scan, root_cause, remediation,
                        each containing the agent's response text
    """
    print("\n[auto-rag] ─── Automated RAG Root-Cause Analysis ───")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
        "analyses": {},
    }

    try:
        from rag.agent import build_agent

        print("[auto-rag] Building agent (llama3.2 via Ollama) ...")
        agent = build_agent(
            llm_backend="ollama",
            model="llama3.2",
            rebuild_kb=rebuild_kb,
        )
        print("[auto-rag] Agent ready.")

        config = {"configurable": {"thread_id": "autonet-auto-analysis"}}

        for query in AUTO_QUERIES:
            qid = query["id"]
            prompt = query["prompt"]
            print("[auto-rag] Running query: %s ..." % qid)

            try:
                result = agent.invoke(
                    {"messages": [("human", prompt)]},
                    config=config,
                )
                response = result["messages"][-1].content
                results["analyses"][qid] = {
                    "query": prompt,
                    "response": response,
                    "status": "success",
                }
                print("[auto-rag]   ✅ %s complete (%d chars)" % (qid, len(response)))

            except Exception as e:
                error_msg = str(e)
                print("[auto-rag]   ❌ %s failed: %s" % (qid, error_msg))
                results["analyses"][qid] = {
                    "query": prompt,
                    "response": "Analysis failed: %s" % error_msg,
                    "status": "error",
                }

        results["status"] = "complete"

    except ImportError as e:
        print("[auto-rag] ⚠ RAG dependencies missing: %s" % e)
        print("[auto-rag]   Generating fallback analysis from data files ...")
        results = _fallback_analysis(results)

    except Exception as e:
        print("[auto-rag] ⚠ RAG agent failed: %s" % e)
        traceback.print_exc()
        results = _fallback_analysis(results)

    # Save report
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, "root_cause_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("[auto-rag] Report saved: %s" % report_path)
    print("[auto-rag] ─── Analysis Complete ───\n")

    return results


def _fallback_analysis(results: dict) -> dict:
    """
    Generate analysis from data files when RAG agent is unavailable.
    Reads anomaly scores, alerts, attack log, and dual-plane report directly.
    """
    results["status"] = "fallback"

    # Load anomaly scores
    scores_path = os.path.join(OUTPUT_DIR, "anomaly_scores.csv")
    alerts_path = os.path.join(OUTPUT_DIR, "alerts.json")
    attack_log_path = os.path.join(OUTPUT_DIR, "attack_log.json")
    dual_plane_path = os.path.join(OUTPUT_DIR, "dual_plane_report.json")

    anomaly_summary = ""
    if os.path.exists(alerts_path):
        with open(alerts_path, "r") as f:
            alerts = json.load(f)
        alert_list = alerts.get("alerts", [])
        anomaly_summary = "ANOMALY SCAN RESULTS\n"
        anomaly_summary += "=" * 40 + "\n"
        anomaly_summary += "Total alerts: %d\n\n" % len(alert_list)
        for a in alert_list:
            anomaly_summary += "Node %d: score=%.4f severity=%s type=%s\n" % (
                a["node_id"], a["anomaly_score"], a["severity"], a["alert_type"]
            )

    root_cause = ""
    if os.path.exists(attack_log_path):
        with open(attack_log_path, "r") as f:
            atk_log = json.load(f)
        root_cause = "ROOT CAUSE ANALYSIS\n"
        root_cause += "=" * 40 + "\n"
        for event in atk_log.get("attacks", []):
            root_cause += "\nAttack: %s (%s)\n" % (event["attack_name"], event["attack_id"])
            root_cause += "Category: %s | Severity: %s\n" % (event["category"], event["severity"])
            root_cause += "MITRE: %s\n" % event.get("mitre_att_ck", "N/A")
            root_cause += "Victims: %s\n" % ", ".join(event["victim_nodes"])
            for ioc in event.get("indicators", []):
                root_cause += "  IoC: %s\n" % ioc

    if os.path.exists(dual_plane_path):
        with open(dual_plane_path, "r") as f:
            dp = json.load(f)
        root_cause += "\nDUAL-PLANE CLASSIFICATION\n"
        root_cause += "-" * 30 + "\n"
        summary = dp.get("summary", {}).get("classifications", {})
        for cls, count in summary.items():
            root_cause += "%s: %d nodes\n" % (cls, count)

    remediation = "REMEDIATION PLAN\n"
    remediation += "=" * 40 + "\n"
    remediation += "1. IMMEDIATE: Isolate flagged nodes from network\n"
    remediation += "2. INVESTIGATE: Review attack logs and IoC indicators\n"
    remediation += "3. VERIFY: Check SSH configs for unauthorized changes\n"
    remediation += "4. HARDEN: Apply security patches and rotate credentials\n"
    remediation += "5. MONITOR: Increase telemetry polling frequency\n"

    results["analyses"] = {
        "anomaly_scan": {"query": "auto", "response": anomaly_summary, "status": "fallback"},
        "root_cause": {"query": "auto", "response": root_cause, "status": "fallback"},
        "remediation": {"query": "auto", "response": remediation, "status": "fallback"},
    }

    return results
