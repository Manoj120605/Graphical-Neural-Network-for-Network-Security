import docker
import time

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

if __name__ == "__main__":
    start_time = time.time()
    active_nodes = discover_nodes()
    results = get_telemetry(active_nodes)
    
    print("\n--- AutoNet-GNN Feature Matrix (Config Drift Plane) ---")
    for r in results:
        status = "🚩 ANOMALY" if r['drift_score'] == 1.0 else "🟢 CLEAN"
        print(f"Node: {r['name']} | Drift Score: {r['drift_score']} | Status: {status}")
    
    print(f"\nScan completed in {round(time.time() - start_time, 2)} seconds.")