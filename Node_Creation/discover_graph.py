import docker

def get_network_topology():
    client = docker.from_env()
    containers = client.containers.list()
    device_list = []

    for container in containers:
        name = container.name
        # Filter for your AutoNet nodes
        if "leaf" in name or "spine" in name:
            # Get the IP address from the backbone_net network
            networks = container.attrs['NetworkSettings']['Networks']
            # Note: This name must match the network in your docker-compose.yml
            ip_addr = networks.get('node_creation_backbone_net', {}).get('IPAddress')
            
            if ip_addr:
                device_list.append({
                    'ip': ip_addr,
                    'user': 'root',        # Default for Alpine nodes
                    'password': 'root',    # You'll need to set this in Alpine
                    'type': 'linux',       # Netmiko type for Alpine
                    'name': name
                })
    
    return device_list

if __name__ == "__main__":
    nodes = get_network_topology()
    print(f"Discovered {len(nodes)} nodes for the AutoNet-GNN Graph.")
    for n in nodes:
        print(f"Node: {n['name']} -> IP: {n['ip']}")