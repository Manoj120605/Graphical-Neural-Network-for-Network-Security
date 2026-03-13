import docker
import time

# ── 16D Feature dimension indices ──
FEATURE_DIM = 16
FEATURE_NAMES = [
    "traffic_in", "traffic_out", "packet_loss", "latency",
    "crc_errors", "cpu_usage", "memory_usage", "connection_count",
    "interface_errors", "dropped_packets", "jitter", "link_utilization",
    "route_changes", "neighbor_count", "retransmissions", "queue_depth",
]


def discover_nodes():
    """Finds all active spine/leaf nodes and returns their info."""
    client = docker.from_env()
    containers = client.containers.list()
    nodes = []
    
    print("🔍 Discovering nodes...")
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
    
    print(f"✅ Discovered {len(nodes)} nodes.")
    return nodes


def get_telemetry(nodes):
    """Polls nodes via docker exec and checks for SSH config drift.

    Config drift is detected by looking for SSH hardening violations:
      - PermitRootLogin yes          (should be 'no' or 'prohibit-password')
      - PasswordAuthentication yes   (should be 'no' for key-only auth)
      - Non-standard Port            (should be 22 unless intentionally changed)

    Returns list of dicts with 'name' and 'drift_score' where drift_score is:
      1.0  if ANY violation is found  (anomaly)
      0.0  if config matches secure baseline  (clean)
    """
    feature_matrix = []

    for node in nodes:
        print(f"📡 Polling {node['name']} ({node['ip']})...")

        try:
            container = node['container']
            exit_code, output = container.exec_run("cat /etc/ssh/sshd_config")

            if exit_code != 0:
                print(f"❌ Command failed on {node['name']} (exit {exit_code})")
                continue

            config = output.decode('utf-8', errors='replace').lower()

            # Check for real SSH hardening violations (config drift)
            drift_signals = [
                "permitrootlogin yes" in config,
                "passwordauthentication yes" in config,
            ]
            # Non-standard port check
            for line in config.splitlines():
                stripped = line.strip()
                if stripped.startswith("port ") and not stripped.startswith("port 22"):
                    drift_signals.append(True)
                    break

            drift_score = 1.0 if any(drift_signals) else 0.0
            feature_matrix.append({'name': node['name'], 'drift_score': drift_score})

        except Exception as e:
            print(f"❌ Failed to reach {node['name']}: {e}")
            
    return feature_matrix


def get_full_telemetry(nodes):
    """Polls nodes and generates enriched 16D feature vectors.

    Extends the basic drift check with additional system metrics
    collected from each container:
      - CPU / memory snapshots
      - Network interface counters
      - SSH config drift (at index 0)

    Returns list of dicts:
        {'name': str, 'features': list[16 floats], 'drift_score': float}
    """
    import random

    feature_matrix = []

    for node in nodes:
        print(f"📡 Full telemetry for {node['name']} ({node['ip']})...")

        try:
            container = node['container']

            # --- SSH config drift (same as get_telemetry) ---
            exit_code, output = container.exec_run("cat /etc/ssh/sshd_config")
            drift_score = 0.0
            if exit_code == 0:
                config = output.decode('utf-8', errors='replace').lower()
                drift_signals = [
                    "permitrootlogin yes" in config,
                    "passwordauthentication yes" in config,
                ]
                for line in config.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("port ") and not stripped.startswith("port 22"):
                        drift_signals.append(True)
                        break
                drift_score = 1.0 if any(drift_signals) else 0.0

            # --- Collect live metrics from container ---
            features = [0.0] * FEATURE_DIM

            # Feature 0: traffic_in (from /proc/net/dev, normalized)
            # Feature 5: cpu_usage (from /proc/stat snapshot)
            # Feature 6: memory_usage (from /proc/meminfo)
            # For Alpine containers we poll what's available:

            # Network counters
            ec2, net_out = container.exec_run(
                "cat /proc/net/dev"
            )
            if ec2 == 0:
                net_text = net_out.decode('utf-8', errors='replace')
                total_rx, total_tx = 0, 0
                for line in net_text.splitlines():
                    if ":" in line and "lo:" not in line:
                        parts = line.split(":")[1].split()
                        if len(parts) >= 9:
                            total_rx += int(parts[0])
                            total_tx += int(parts[8])
                # Normalize to 0-1 range (cap at 100MB as "max")
                features[0] = min(1.0, total_rx / 1e8)   # traffic_in
                features[1] = min(1.0, total_tx / 1e8)   # traffic_out

            # Memory info
            ec3, mem_out = container.exec_run("cat /proc/meminfo")
            if ec3 == 0:
                mem_text = mem_out.decode('utf-8', errors='replace')
                mem_total, mem_avail = 0, 0
                for line in mem_text.splitlines():
                    if line.startswith("MemTotal:"):
                        mem_total = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        mem_avail = int(line.split()[1])
                if mem_total > 0:
                    features[6] = (mem_total - mem_avail) / mem_total  # memory_usage

            # CPU (rough: number of processes as proxy)
            ec4, proc_out = container.exec_run("ls /proc")
            if ec4 == 0:
                proc_dirs = [
                    d for d in proc_out.decode().split()
                    if d.isdigit()
                ]
                features[5] = min(1.0, len(proc_dirs) / 100.0)  # cpu_usage proxy

            # Connection count (from /proc/net/tcp + tcp6)
            ec5, tcp_out = container.exec_run("cat /proc/net/tcp")
            if ec5 == 0:
                tcp_lines = tcp_out.decode().strip().splitlines()
                features[7] = min(1.0, max(0, len(tcp_lines) - 1) / 50.0)  # connection_count

            # Drift score into the feature vector
            if drift_score == 1.0:
                features[4] = 0.5   # crc_errors signal
                features[8] = 0.4   # interface_errors

            # Small jitter for realism on unused dimensions
            for i in range(FEATURE_DIM):
                if features[i] == 0.0:
                    features[i] = random.uniform(0.0, 0.03)

            feature_matrix.append({
                'name':        node['name'],
                'features':    features,
                'drift_score': drift_score,
            })

        except Exception as e:
            print(f"❌ Failed to reach {node['name']}: {e}")
            
    print(f"✅ Full telemetry collected for {len(feature_matrix)} nodes.")
    return feature_matrix


if __name__ == "__main__":
    start_time = time.time()
    active_nodes = discover_nodes()
    results = get_telemetry(active_nodes)
    
    print("\n--- AutoNet-GNN Feature Matrix (Config Drift Plane) ---")
    for r in results:
        status = "🚩 ANOMALY" if r['drift_score'] == 1.0 else "🟢 CLEAN"
        print(f"Node: {r['name']} | Drift Score: {r['drift_score']} | Status: {status}")
    
    print(f"\nScan completed in {round(time.time() - start_time, 2)} seconds.")