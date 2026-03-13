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
    """Polls nodes via docker exec and checks for 'Config Drift' (Telnet enabled)."""
    feature_matrix = []
    
    for node in nodes:
        print(f"📡 Polling {node['name']} ({node['ip']})...")
        
        try:
            container = node['container']
            exit_code, output = container.exec_run("cat /etc/ssh/sshd_config")
            
            if exit_code != 0:
                print(f"❌ Command failed on {node['name']} (exit {exit_code})")
                continue
            
            config = output.decode('utf-8', errors='replace')
            
            # Logic: 1.0 if 'telnet' is found (Anomaly), else 0.0 (Clean)
            telnet_drift = 1.0 if "telnet" in config.lower() else 0.0
            feature_matrix.append([node['name'], telnet_drift])
                
        except Exception as e:
            print(f"❌ Failed to reach {node['name']}: {e}")
            
    return feature_matrix

if __name__ == "__main__":
    start_time = time.time()
    active_nodes = discover_nodes()
    results = get_telemetry(active_nodes)
    
    print("\n--- AutoNet-GNN Feature Matrix (Config Drift Plane) ---")
    for r in results:
        status = "🚩 ANOMALY" if r[1] == 1.0 else "🟢 CLEAN"
        print(f"Node: {r[0]} | Drift Score: {r[1]} | Status: {status}")
    
    print(f"\nScan completed in {round(time.time() - start_time, 2)} seconds.")