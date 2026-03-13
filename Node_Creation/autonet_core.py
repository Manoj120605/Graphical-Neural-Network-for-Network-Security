"""
autonet_core.py
===============
Discovers Docker spine/leaf containers and collects real multi-dimensional
telemetry via ``docker exec``, producing a feature matrix suitable for the
GraphSAGE anomaly detection model.

Telemetry dimensions (16, matching SYSTEM_DESIGN Section 4):
    0  traffic_in          - rx bytes from /proc/net/dev (eth0)
    1  traffic_out         - tx bytes from /proc/net/dev (eth0)
    2  packet_loss         - rx_drop + tx_drop
    3  latency             - (reserved, 0.0)
    4  crc_errors          - rx_errs + tx_errs from /proc/net/dev
    5  cpu_usage           - from /proc/stat (1 - idle ratio)
    6  memory_usage        - from /proc/meminfo (1 - free/total)
    7  connection_count    - line count from /proc/net/tcp + tcp6
    8  interface_errors     - total errors across all interfaces
    9  dropped_packets     - total drops across all interfaces
   10  jitter              - (reserved, 0.0)
   11  link_utilization    - (reserved, 0.0)
   12  route_changes       - line count from /proc/net/route - 1
   13  neighbor_count      - line count from /proc/net/arp - 1
   14  retransmissions     - (reserved, 0.0)
   15  config_drift        - SSH hardening violations (0 or 1)
"""

import docker
import time


def discover_nodes():
    """Finds all active spine/leaf containers and returns their info."""
    client = docker.from_env()
    containers = client.containers.list()
    nodes = []

    print("[discover] Scanning Docker containers...")
    for container in containers:
        name = container.name
        if "leaf" in name or "spine" in name:
            networks = container.attrs['NetworkSettings']['Networks']
            ip_addr = networks.get('node_creation_backbone_net', {}).get('IPAddress')

            if ip_addr:
                nodes.append({
                    'ip': ip_addr,
                    'name': name,
                    'container': container,
                })

    print("[discover] Found %d nodes." % len(nodes))
    return nodes


def _parse_net_dev(output: str) -> dict:
    """Parse /proc/net/dev and return per-interface stats."""
    stats = {}
    for line in output.strip().splitlines()[2:]:  # skip header lines
        parts = line.split()
        if len(parts) < 11:
            continue
        iface = parts[0].rstrip(":")
        stats[iface] = {
            "rx_bytes": float(parts[1]),
            "rx_errs": float(parts[3]),
            "rx_drop": float(parts[4]),
            "tx_bytes": float(parts[9]),
            "tx_errs": float(parts[11]) if len(parts) > 11 else 0.0,
            "tx_drop": float(parts[12]) if len(parts) > 12 else 0.0,
        }
    return stats


def _parse_cpu(output: str) -> float:
    """Parse /proc/stat first line and return CPU usage ratio (0-1)."""
    for line in output.strip().splitlines():
        if line.startswith("cpu "):
            parts = line.split()
            if len(parts) >= 5:
                user, nice, system, idle = (float(parts[1]), float(parts[2]),
                                            float(parts[3]), float(parts[4]))
                total = user + nice + system + idle
                if total > 0:
                    return 1.0 - (idle / total)
    return 0.0


def _parse_meminfo(output: str) -> float:
    """Parse /proc/meminfo and return memory usage ratio (0-1)."""
    mem = {}
    for line in output.strip().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            key = parts[0].rstrip(":")
            mem[key] = float(parts[1])
    total = mem.get("MemTotal", 1)
    free = mem.get("MemAvailable", mem.get("MemFree", 0))
    if total > 0:
        return 1.0 - (free / total)
    return 0.0


def _check_ssh_drift(config: str) -> float:
    """Check SSH config for hardening violations. Returns 0.0 or 1.0."""
    config_lower = config.lower()
    drift_signals = [
        "permitrootlogin yes" in config_lower,
        "passwordauthentication yes" in config_lower,
    ]
    for line in config_lower.splitlines():
        stripped = line.strip()
        if stripped.startswith("port ") and not stripped.startswith("port 22"):
            drift_signals.append(True)
            break
    return 1.0 if any(drift_signals) else 0.0


def _exec_cmd(container, cmd: str) -> str:
    """Execute a command in a container and return stdout, or empty string on failure."""
    try:
        exit_code, output = container.exec_run(cmd)
        if exit_code == 0:
            return output.decode("utf-8", errors="replace")
    except Exception:
        pass
    return ""


def _count_lines(text: str) -> int:
    """Count non-empty lines."""
    return len([l for l in text.strip().splitlines() if l.strip()])


def get_telemetry(nodes):
    """Poll containers via docker exec and collect real multi-dimensional telemetry.

    Returns list of [node_name, [16-dim feature vector]].
    """
    feature_matrix = []

    for node in nodes:
        print("[telemetry] Polling %s (%s)..." % (node['name'], node['ip']))

        container = node['container']

        try:
            # Network stats from /proc/net/dev
            net_dev = _exec_cmd(container, "cat /proc/net/dev")
            net_stats = _parse_net_dev(net_dev)
            # Prefer eth0, fall back to first non-lo interface
            iface_stats = net_stats.get("eth0", {})
            if not iface_stats:
                for k, v in net_stats.items():
                    if k != "lo":
                        iface_stats = v
                        break
            if not iface_stats:
                iface_stats = {"rx_bytes": 0, "tx_bytes": 0, "rx_errs": 0,
                               "tx_errs": 0, "rx_drop": 0, "tx_drop": 0}

            # CPU usage
            cpu_stat = _exec_cmd(container, "cat /proc/stat")
            cpu_usage = _parse_cpu(cpu_stat)

            # Memory usage
            meminfo = _exec_cmd(container, "cat /proc/meminfo")
            mem_usage = _parse_meminfo(meminfo)

            # Connection count (TCP + TCP6)
            tcp_lines = _exec_cmd(container, "cat /proc/net/tcp")
            tcp6_lines = _exec_cmd(container, "cat /proc/net/tcp6")
            conn_count = max(0, _count_lines(tcp_lines) - 1) + max(0, _count_lines(tcp6_lines) - 1)

            # Total interface errors/drops across all interfaces
            total_errors = sum(s.get("rx_errs", 0) + s.get("tx_errs", 0) for s in net_stats.values() if s)
            total_drops = sum(s.get("rx_drop", 0) + s.get("tx_drop", 0) for s in net_stats.values() if s)

            # Route count
            routes = _exec_cmd(container, "cat /proc/net/route")
            route_count = max(0, _count_lines(routes) - 1)

            # ARP neighbor count
            arp = _exec_cmd(container, "cat /proc/net/arp")
            arp_count = max(0, _count_lines(arp) - 1)

            # SSH config drift
            ssh_config = _exec_cmd(container, "cat /etc/ssh/sshd_config")
            drift = _check_ssh_drift(ssh_config)

            # Build 16D feature vector
            features = [
                iface_stats.get("rx_bytes", 0.0),       # 0  traffic_in
                iface_stats.get("tx_bytes", 0.0),       # 1  traffic_out
                iface_stats.get("rx_drop", 0.0) + iface_stats.get("tx_drop", 0.0),  # 2  packet_loss
                0.0,                                     # 3  latency (reserved)
                iface_stats.get("rx_errs", 0.0) + iface_stats.get("tx_errs", 0.0),  # 4  crc_errors
                cpu_usage,                               # 5  cpu_usage
                mem_usage,                               # 6  memory_usage
                float(conn_count),                       # 7  connection_count
                total_errors,                            # 8  interface_errors
                total_drops,                             # 9  dropped_packets
                0.0,                                     # 10 jitter (reserved)
                0.0,                                     # 11 link_utilization (reserved)
                float(route_count),                      # 12 route_changes
                float(arp_count),                        # 13 neighbor_count
                0.0,                                     # 14 retransmissions (reserved)
                drift,                                   # 15 config_drift
            ]

            feature_matrix.append([node['name'], features])

        except Exception as e:
            print("[telemetry] FAILED %s: %s" % (node['name'], e))

    return feature_matrix


if __name__ == "__main__":
    start_time = time.time()
    active_nodes = discover_nodes()
    results = get_telemetry(active_nodes)

    print("\n--- AutoNet-GNN Feature Matrix (16D Telemetry) ---")
    for r in results:
        name = r[0]
        feats = r[1]
        drift_flag = " DRIFT" if feats[15] == 1.0 else ""
        print("  %s | rx=%.0f tx=%.0f cpu=%.2f mem=%.2f conns=%d%s"
              % (name, feats[0], feats[1], feats[5], feats[6], int(feats[7]), drift_flag))

    print("\nScan completed in %.2f seconds." % (time.time() - start_time))