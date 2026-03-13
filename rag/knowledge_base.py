import os
import json
import numpy as np
import torch
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GNN'))

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # free, runs locally


def build_knowledge_base(data_path="synthetic_graph.pt",
                         scores_path="anomaly_scores.csv") -> Chroma:
    """
    Build a vector store from:
    - Per-node anomaly reports (from GNN outputs)
    - Network topology context
    - Security knowledge snippets
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    documents = []
    
    # 1. Load GNN graph data
    if os.path.exists(data_path):
        data = torch.load(data_path, weights_only=False)
        
        # Load anomaly scores if available
        scores = None
        if os.path.exists(scores_path):
            raw = np.loadtxt(scores_path, delimiter=",", skiprows=1)
            scores = {int(r[0]): (r[1], int(r[2])) for r in raw}
        
        mean_score = np.mean([s[0] for s in scores.values()]) if scores else 0
        std_score  = np.std([s[0] for s in scores.values()]) if scores else 1
        
        # Create a document per node
        for node_id in range(data.num_nodes):
            features = data.x[node_id].numpy().tolist()
            label    = data.y[node_id].item()
            
            score_val = scores[node_id][0] if scores and node_id in scores else 0
            sigma     = (score_val - mean_score) / (std_score + 1e-8)
            
            # Get neighbors from edge_index
            mask      = data.edge_index[0] == node_id
            neighbors = data.edge_index[1][mask].tolist()
            
            content = f"""
Node ID: {node_id}
Label: {"ANOMALY" if label == 1 else "NORMAL"}
Anomaly Score: {score_val:.4f} ({sigma:.2f}σ)
Degree (connections): {len(neighbors)}
Neighbors: {neighbors[:10]}  
Feature mean: {np.mean(features):.4f}
Feature std: {np.std(features):.4f}
Feature max: {max(features):.4f}
Feature min: {min(features):.4f}
Status: {"CRITICAL - confirmed anomaly" if label == 1 else "Normal operation"}
"""
            documents.append(Document(
                page_content=content,
                metadata={
                    "node_id": node_id,
                    "label": label,
                    "score": score_val,
                    "sigma": sigma,
                    "type": "node_report"
                }
            ))
    
    # 2. Add security knowledge base (static docs for RAG context)
    security_docs = [
        Document(page_content="""
Config Drift Detection: When a node's feature vector deviates significantly 
from its neighborhood mean, it indicates unauthorized configuration changes.
Remediation: Roll back to last known-good config. Check change management logs.
Typical causes: Unauthorized ACL modification, SNMP community string changes, 
NTP server removal, routing table manipulation.
""", metadata={"type": "knowledge", "topic": "config_drift"}),

        Document(page_content="""
Lateral Movement Detection: Nodes with unusually high anomaly scores in 
access-tier roles may indicate East-West attack propagation.
Remediation: Isolate node, capture traffic, audit authentication logs.
Typical indicators: Abnormal connection patterns to core nodes, 
unusual protocol usage, off-hours activity.
""", metadata={"type": "knowledge", "topic": "lateral_movement"}),

        Document(page_content="""
Silent Hardware Degradation: Nodes with gradually increasing anomaly scores
over multiple scan cycles may indicate hardware failure below alarm threshold.
Remediation: Schedule maintenance window, check interface CRC counters,
verify power supply and temperature sensors.
""", metadata={"type": "knowledge", "topic": "hardware_degradation"}),

        Document(page_content="""
Graph Poisoning: If multiple high-degree hub nodes simultaneously show 
anomaly spikes, this may indicate a coordinated monitoring attack.
Remediation: Activate dual-plane verification, compare physical LLDP tables
against logical topology, isolate management plane.
""", metadata={"type": "knowledge", "topic": "graph_poisoning"}),
    ]
    documents.extend(security_docs)
    
    # Build or load Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    
    print(f"✔ Knowledge base built: {len(documents)} documents indexed")
    return vectorstore


def load_knowledge_base() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
