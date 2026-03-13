"""
attack_simulator.py
====================
Synthetic attack simulation engine for AutoNet-GNN.

Reads attack pattern definitions from  synthetic-attacks/*.json  and
generates enriched 16-dimensional feature vectors that encode the
network-level effects each attack would produce.

The 16 feature dimensions are (matching FEATURE_NAMES in rag/tools.py):
    0  traffic_in          1  traffic_out         2  packet_loss
    3  latency             4  crc_errors          5  cpu_usage
    6  memory_usage        7  connection_count    8  interface_errors
    9  dropped_packets    10  jitter             11  link_utilization
   12  route_changes      13  neighbor_count     14  retransmissions
   15  queue_depth

IMPORTANT: This module is for SIMULATION / EDUCATIONAL use only.
           It does NOT execute real attacks against any infrastructure.
"""

import json
import os
import random
import time
from datetime import datetime, timezone
from typing import Any

# ── Feature dimension indices ─────────────────────────────────────────
TRAFFIC_IN       = 0
TRAFFIC_OUT      = 1
PACKET_LOSS      = 2
LATENCY          = 3
CRC_ERRORS       = 4
CPU_USAGE        = 5
MEMORY_USAGE     = 6
CONNECTION_COUNT = 7
INTERFACE_ERRORS = 8
DROPPED_PACKETS  = 9
JITTER           = 10
LINK_UTILIZATION = 11
ROUTE_CHANGES    = 12
NEIGHBOR_COUNT   = 13
RETRANSMISSIONS  = 14
QUEUE_DEPTH      = 15

FEATURE_DIM = 16

# ── Normal node baseline (low / idle values) ──────────────────────────
NORMAL_BASELINE = [
    0.05,   # traffic_in         (5% utilization)
    0.03,   # traffic_out
    0.01,   # packet_loss        (1%)
    0.02,   # latency            (low)
    0.00,   # crc_errors
    0.08,   # cpu_usage          (8%)
    0.12,   # memory_usage       (12%)
    0.04,   # connection_count   (low)
    0.00,   # interface_errors
    0.01,   # dropped_packets
    0.01,   # jitter
    0.06,   # link_utilization
    0.00,   # route_changes
    0.05,   # neighbor_count     (normal)
    0.01,   # retransmissions
    0.02,   # queue_depth
]

# ═══════════════════════════════════════════════════════════════════════
# ATTACK FEATURE PROFILES
# ═══════════════════════════════════════════════════════════════════════
# Each profile maps attack category -> dict of {feature_index: (min, max)}
# Values represent the ANOMALOUS level (will be drawn uniformly)

ATTACK_PROFILES: dict[str, dict[int, tuple[float, float]]] = {
    # ATK-001: DDoS Pulse Wave
    "ddos_pulse_wave": {
        TRAFFIC_IN:       (0.80, 1.00),
        TRAFFIC_OUT:      (0.10, 0.25),
        PACKET_LOSS:      (0.30, 0.60),
        LATENCY:          (0.50, 0.80),
        CONNECTION_COUNT: (0.85, 1.00),
        DROPPED_PACKETS:  (0.40, 0.70),
        JITTER:           (0.30, 0.60),
        LINK_UTILIZATION: (0.80, 1.00),
        QUEUE_DEPTH:      (0.70, 0.95),
    },

    # ATK-002: DNS Hijacking
    "dns_hijacking": {
        TRAFFIC_OUT:      (0.40, 0.70),
        LATENCY:          (0.20, 0.45),
        CONNECTION_COUNT: (0.50, 0.75),
        ROUTE_CHANGES:    (0.60, 0.90),
        INTERFACE_ERRORS: (0.10, 0.30),
        RETRANSMISSIONS:  (0.15, 0.35),
    },

    # ATK-003: Network Reconnaissance
    "network_recon": {
        TRAFFIC_IN:       (0.15, 0.35),
        TRAFFIC_OUT:      (0.20, 0.40),
        CONNECTION_COUNT: (0.70, 0.95),
        DROPPED_PACKETS:  (0.20, 0.40),
        RETRANSMISSIONS:  (0.30, 0.55),
        LINK_UTILIZATION: (0.25, 0.45),
    },

    # ATK-004: SSH Brute Force
    "ssh_brute_force": {
        CONNECTION_COUNT: (0.75, 1.00),
        CPU_USAGE:        (0.40, 0.65),
        RETRANSMISSIONS:  (0.35, 0.60),
        LATENCY:          (0.15, 0.35),
        DROPPED_PACKETS:  (0.10, 0.25),
    },

    # ATK-005: SQL Injection
    "sql_injection": {
        TRAFFIC_OUT:      (0.55, 0.85),
        CPU_USAGE:        (0.50, 0.80),
        MEMORY_USAGE:     (0.45, 0.70),
        CONNECTION_COUNT: (0.30, 0.55),
        LATENCY:          (0.20, 0.40),
    },

    # ATK-006: ARP Spoofing / MITM
    "arp_spoofing": {
        TRAFFIC_IN:       (0.35, 0.60),
        TRAFFIC_OUT:      (0.35, 0.60),
        INTERFACE_ERRORS: (0.50, 0.80),
        NEIGHBOR_COUNT:   (0.60, 0.90),
        CRC_ERRORS:       (0.15, 0.35),
        ROUTE_CHANGES:    (0.30, 0.55),
    },

    # ATK-007: Ransomware
    "ransomware": {
        CPU_USAGE:        (0.75, 1.00),
        MEMORY_USAGE:     (0.60, 0.90),
        TRAFFIC_OUT:      (0.40, 0.65),  # C2 beacon
        LINK_UTILIZATION: (0.50, 0.75),
        DROPPED_PACKETS:  (0.15, 0.30),
    },

    # ATK-008: HTTP Slowloris
    "http_slowloris": {
        CONNECTION_COUNT: (0.90, 1.00),
        TRAFFIC_IN:       (0.60, 0.85),
        LATENCY:          (0.65, 0.90),
        QUEUE_DEPTH:      (0.80, 1.00),
        PACKET_LOSS:      (0.20, 0.40),
        JITTER:           (0.25, 0.50),
    },

    # ATK-009: SMB Pass-the-Hash
    "smb_pass_the_hash": {
        CONNECTION_COUNT: (0.55, 0.80),
        TRAFFIC_OUT:      (0.30, 0.55),
        LATENCY:          (0.20, 0.40),
        CPU_USAGE:        (0.25, 0.45),
        RETRANSMISSIONS:  (0.20, 0.35),
    },

    # ATK-010: DNS Tunneling Exfiltration
    "dns_tunneling": {
        TRAFFIC_OUT:      (0.65, 0.95),
        CONNECTION_COUNT: (0.60, 0.85),
        LATENCY:          (0.15, 0.30),
        RETRANSMISSIONS:  (0.25, 0.45),
        LINK_UTILIZATION: (0.40, 0.65),
    },
}

# Map attack JSON "id" or "simulation.pattern" to our profile key
_PATTERN_MAP: dict[str, str] = {
    "ATK-001": "ddos_pulse_wave",
    "ATK-002": "dns_hijacking",
    "ATK-003": "network_recon",
    "ATK-004": "ssh_brute_force",
    "ATK-005": "sql_injection",
    "ATK-006": "arp_spoofing",
    "ATK-007": "ransomware",
    "ATK-008": "http_slowloris",
    "ATK-009": "smb_pass_the_hash",
    "ATK-010": "dns_tunneling",
    # Also allow matching by simulation.pattern field:
    "pulse_wave":              "ddos_pulse_wave",
    "dns_spoof":               "dns_hijacking",
    "progressive_sweep":       "network_recon",
    "dictionary_brute_force":  "ssh_brute_force",
    "union_based_sqli":        "sql_injection",
    "bidirectional_arp_poison": "arp_spoofing",
    "ransomware_sim":          "ransomware",
    "slow_headers":            "http_slowloris",
    "pass_the_hash":           "smb_pass_the_hash",
    "dns_tunnel_exfil":        "dns_tunneling",
}


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

def load_attack_patterns(attack_dir: str) -> list[dict]:
    """
    Load all ATK-*.json files from the given directory.

    Returns a list of parsed attack pattern dictionaries.
    """
    patterns = []
    if not os.path.isdir(attack_dir):
        print("[attack_sim] WARNING: attack directory not found: %s" % attack_dir)
        return patterns

    for fname in sorted(os.listdir(attack_dir)):
        if fname.startswith("ATK-") and fname.endswith(".json"):
            fpath = os.path.join(attack_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                pattern = json.load(f)
            patterns.append(pattern)

    print("[attack_sim] Loaded %d attack patterns from %s" % (len(patterns), attack_dir))
    return patterns


def select_attack_scenarios(
    patterns: list[dict],
    count: int = 3,
    seed: int | None = None,
) -> list[dict]:
    """
    Randomly select `count` attack patterns to simulate.

    Args:
        patterns: Full list of loaded attack patterns.
        count:    How many distinct attacks to simulate (capped at len(patterns)).
        seed:     Optional RNG seed for reproducibility.

    Returns:
        List of selected attack pattern dicts.
    """
    if seed is not None:
        random.seed(seed)
    count = min(count, len(patterns))
    selected = random.sample(patterns, k=count)
    print("[attack_sim] Selected %d attacks: %s" % (
        len(selected),
        ", ".join(p.get("id", "???") for p in selected),
    ))
    return selected


def _resolve_profile(pattern: dict) -> str | None:
    """Resolve an attack pattern dict to a profile key."""
    atk_id = pattern.get("id", "")
    if atk_id in _PATTERN_MAP:
        return _PATTERN_MAP[atk_id]
    sim_pattern = pattern.get("simulation", {}).get("pattern", "")
    if sim_pattern in _PATTERN_MAP:
        return _PATTERN_MAP[sim_pattern]
    return None


def _generate_attack_features(profile_key: str) -> list[float]:
    """
    Generate a 16D feature vector for an attacked node using the
    specified attack profile.  Non-attack dimensions use the normal
    baseline with small random jitter.
    """
    profile = ATTACK_PROFILES.get(profile_key, {})
    features = list(NORMAL_BASELINE)

    # Apply normal jitter to all dimensions first
    for i in range(FEATURE_DIM):
        jitter = random.uniform(-0.02, 0.02)
        features[i] = max(0.0, min(1.0, features[i] + jitter))

    # Overlay attack-specific anomalous values
    for dim_idx, (lo, hi) in profile.items():
        features[dim_idx] = random.uniform(lo, hi)

    return features


def _generate_normal_features() -> list[float]:
    """Generate a 16D feature vector for a clean (non-attacked) node."""
    features = list(NORMAL_BASELINE)
    for i in range(FEATURE_DIM):
        jitter = random.uniform(-0.02, 0.02)
        features[i] = max(0.0, min(1.0, features[i] + jitter))
    return features


def generate_attack_telemetry(
    node_names: list[str],
    scenarios: list[dict],
    victims_per_attack: int = 2,
    seed: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Generate enriched 16D telemetry for all nodes, injecting attack
    signatures into randomly selected victim nodes.

    Args:
        node_names:          Ordered list of container names.
        scenarios:           Attack patterns selected for simulation.
        victims_per_attack:  Number of victim nodes per attack.
        seed:                Optional RNG seed.

    Returns:
        (telemetry, attack_log)

        telemetry:  list of dicts with keys:
            'name'        : str   — container name
            'features'    : list[float]  — 16D feature vector
            'drift_score' : float — 1.0 if attacked, 0.0 otherwise
            'attack_id'   : str | None — attack ID if this node is a victim

        attack_log: list of dicts describing each simulated attack event
    """
    if seed is not None:
        random.seed(seed)

    # Track which nodes are victims of which attack
    victim_map: dict[str, dict] = {}      # node_name -> attack info
    attack_log: list[dict] = []

    leaf_nodes = [n for n in node_names if "leaf" in n.lower()]
    spine_nodes = [n for n in node_names if "spine" in n.lower()]

    for pattern in scenarios:
        profile_key = _resolve_profile(pattern)
        if profile_key is None:
            print("[attack_sim] WARNING: no profile for %s, skipping" %
                  pattern.get("id", "???"))
            continue

        # Select victim nodes (prefer leaf nodes as targets)
        available = [n for n in leaf_nodes if n not in victim_map]
        if len(available) < victims_per_attack:
            available += [n for n in spine_nodes if n not in victim_map]

        count = min(victims_per_attack, len(available))
        if count == 0:
            print("[attack_sim] WARNING: no available victim nodes for %s" %
                  pattern.get("id", "???"))
            continue

        victims = random.sample(available, k=count)

        for victim in victims:
            victim_map[victim] = {
                "attack_id":   pattern.get("id", "???"),
                "attack_name": pattern.get("name", "Unknown"),
                "category":    pattern.get("category", "Unknown"),
                "severity":    pattern.get("severity", "unknown"),
                "mitre":       pattern.get("mitre_att_ck", ""),
                "profile_key": profile_key,
            }

        attack_log.append({
            "attack_id":    pattern.get("id", "???"),
            "attack_name":  pattern.get("name", "Unknown"),
            "category":     pattern.get("category", "Unknown"),
            "severity":     pattern.get("severity", "unknown"),
            "mitre_att_ck": pattern.get("mitre_att_ck", ""),
            "profile_key":  profile_key,
            "victim_nodes": victims,
            "indicators":   pattern.get("indicators_of_compromise", []),
            "detection_rules": pattern.get("detection_rules", {}),
            "timestamp":    datetime.now(timezone.utc).isoformat(),
        })

    # Build telemetry for ALL nodes
    telemetry: list[dict] = []
    for name in node_names:
        if name in victim_map:
            info = victim_map[name]
            features = _generate_attack_features(info["profile_key"])
            telemetry.append({
                "name":        name,
                "features":    features,
                "drift_score": 1.0,
                "attack_id":   info["attack_id"],
                "attack_name": info["attack_name"],
                "category":    info["category"],
                "severity":    info["severity"],
                "mitre":       info["mitre"],
            })
        else:
            features = _generate_normal_features()
            telemetry.append({
                "name":        name,
                "features":    features,
                "drift_score": 0.0,
                "attack_id":   None,
            })

    # Summary
    attacked = sum(1 for t in telemetry if t["drift_score"] == 1.0)
    print("[attack_sim] Telemetry generated: %d nodes (%d attacked, %d clean)"
          % (len(telemetry), attacked, len(telemetry) - attacked))

    return telemetry, attack_log


def simulate_attacks(
    node_names: list[str],
    attack_dir: str,
    num_attacks: int = 3,
    victims_per_attack: int = 2,
    seed: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Full attack simulation pipeline:
      1. Load attack patterns from disk
      2. Select random subset of attacks
      3. Generate enriched telemetry with injected anomalies

    Args:
        node_names:          Ordered list of container names.
        attack_dir:          Path to  synthetic-attacks/  directory.
        num_attacks:         Number of distinct attacks to simulate.
        victims_per_attack:  Number of victim nodes per attack.
        seed:                Optional RNG seed for reproducibility.

    Returns:
        (telemetry, attack_log)  — see generate_attack_telemetry docs
    """
    print("\n[attack_sim] ─── Starting Attack Simulation ───")
    start = time.time()

    patterns = load_attack_patterns(attack_dir)
    if not patterns:
        raise RuntimeError(
            "No attack patterns found in %s. "
            "Ensure synthetic-attacks/ATK-*.json files exist." % attack_dir
        )

    scenarios = select_attack_scenarios(patterns, count=num_attacks, seed=seed)
    telemetry, attack_log = generate_attack_telemetry(
        node_names, scenarios,
        victims_per_attack=victims_per_attack,
        seed=seed,
    )

    elapsed = round(time.time() - start, 2)
    print("[attack_sim] ─── Simulation Complete (%.2fs) ───\n" % elapsed)

    return telemetry, attack_log


def save_attack_log(attack_log: list[dict], output_dir: str) -> str:
    """Save the attack log to a JSON file for RAG indexing."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "attack_log.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "simulation_timestamp": datetime.now(timezone.utc).isoformat(),
            "attacks": attack_log,
            "total_attacks": len(attack_log),
        }, f, indent=2)
    print("[attack_sim] Saved attack log: %s" % path)
    return path


# ═══════════════════════════════════════════════════════════════════════
# Standalone test / demo
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Quick demo with mock node names
    mock_names = (
        ["node_creation-spine-%d" % i for i in range(1, 5)]
        + ["node_creation-leaf-%d" % i for i in range(1, 37)]
    )

    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    atk_dir = os.path.join(_PROJECT_ROOT, "synthetic-attacks")

    telemetry, log = simulate_attacks(
        mock_names, atk_dir,
        num_attacks=3, victims_per_attack=2, seed=42,
    )

    print("\n--- Attack Log ---")
    for entry in log:
        print("  %s: %s -> victims: %s" % (
            entry["attack_id"], entry["attack_name"],
            entry["victim_nodes"],
        ))

    print("\n--- Telemetry Sample (first 5 nodes) ---")
    for t in telemetry[:5]:
        status = "🚩 %s" % t.get("attack_name", "") if t["drift_score"] == 1.0 else "🟢 CLEAN"
        print("  %-30s | drift=%.1f | %s" % (t["name"], t["drift_score"], status))
        print("    features: [%s]" % ", ".join("%.3f" % f for f in t["features"]))

    save_attack_log(log, os.path.join(_PROJECT_ROOT, "syntheticdata"))
