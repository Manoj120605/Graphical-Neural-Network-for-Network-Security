import docker
from netmiko import ConnectHandler
import time

def discover_nodes():
    """Finds all 45 nodes and gets their IP addresses."""
    client = docker.from_env()
    containers = client.containers.list()
    nodes = []
    
    print("🔍 Discovering nodes...")
    for container in containers:
        name = container.name
        if "leaf" in name or "spine" in name:
            # Matches the network name created by your docker-compose
            networks = container.attrs['NetworkSettings']['Networks']
            ip_addr = networks.get('codetocareear_backbone_net', {}).get('IPAddress')
            
            if ip_addr:
                nodes.append({'ip': ip_addr, 'name': name})
    
    print(f"✅ Discovered {len(nodes)} nodes.")
    return nodes

def get_telemetry(nodes):
    """Logs into nodes via SSH and checks for 'Config Drift' (Telnet enabled)."""
    feature_matrix = []
    
    for node in nodes:
        print(f"📡 Polling {node['name']} ({node['ip']})...")
        device = {
            'device_type': 'linux', # Netmiko type for Alpine
            'host': node['ip'],
            'username': 'root',
            'password': 'root',
            'port': 22,
        }
        
        try:
            with ConnectHandler(**device) as ssh:
                # We check the SSH config to see if Telnet (insecure) was injected
                config = ssh.send_command("cat /etc/ssh/sshd_config")
                
                # Logic: 1.0 if 'telnet' is found (Anomaly), else 0.0 (Clean) [cite: 9]
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