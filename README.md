# AutoNet-GNN
### Cognitive Network Resilience Platform

> *"This is not the next SIEM. This is the network that defends itself."*

![AutoNet-GNN System Architecture](autonet_arch.png)

---

## What the Project Does

AutoNet-GNN is a **cognitive network security platform** that models enterprise infrastructure as a live, continuously evolving graph. By applying Graph Neural Networks (GNNs), temporal modeling, and an agentic AI reasoning layer, it detects threats, anomalies, and structural failures that traditional threshold-based security tools fundamentally cannot see.

The system ingests real-time telemetry from multiple sources — SNMP/Syslog, Netmiko SSH, IPFIX/NetFlow, and BGP/LLDP — and transforms raw network state into a dynamic graph. A layered ML pipeline then performs:

- **Spatial anomaly detection** via GraphSAGE / GAT message passing across device neighborhoods
- **Temporal drift detection** via TGN / LSTM, tracking how node behavior evolves over time
- **Dual-Plane Verification** that cross-references physical-layer state against logical-plane topology to prune adversarial graph injections
- **Agentic root-cause synthesis** via LangChain, backed by a CVE/runbook RAG knowledge base
- **Explainable AI (XAI)** that translates GNN matrix outputs into human-readable repair plans
- **On-premises CUDA inference** on an RTX 5060 / Intel Ultra 9 — zero data egress

Operators interact through a Human-in-the-Loop approval gate: the system detects and proposes, but a human must confirm before any remediation is applied.

### Threats Addressed

| Threat | Legacy Blind Spot | AutoNet-GNN Approach |
|---|---|---|
| Pulse-Wave DDoS | Stays under threshold | Spatio-temporal graph flow + LSTM drift |
| Lateral Movement | Inside "trusted" VLAN | GNN neighborhood anomaly scoring |
| BGP / DNS Hijacking | Slow manual detection | Agentic RAG vs. historical graph baseline |
| Config Drift | Hours-long audit gaps | Continuous GNN edge-state comparison |
| Silent Hardware Degradation | Below alarm threshold | Spatial entropy Z-score vs. neighbors |
| Graph / Structural Poisoning | Trusts compromised monitor | Dual-Plane cross-reference + auto-pruning |

---

## Why the Project is Useful

Enterprise security today is **reactive and fragmented** — built on static thresholds that sophisticated adversaries trivially bypass. AutoNet-GNN represents a fundamental shift from scalar alerting to relational reasoning.

### Key Business Advantages

**500 Alerts → 1 Root-Cause Report**
When a core switch fails, legacy SIEMs generate one alert per downstream dependent. AutoNet-GNN understands parent-child graph topology and collapses the entire alert storm into a single diagnostic, directly reducing Mean Time To Resolve (MTTR) and eliminating operator fatigue.

**Sub-threshold Silent Failure Detection**
Hardware degrading gradually — CRC errors, intermittent link flaps — often stays below alarm thresholds for hours before total failure. AutoNet-GNN's Z-score entropy model compares each node's metric distribution to its spatial neighborhood. A port that is technically "Up" but statistically diverging from its neighbors gets flagged as an early warning, 30–60 minutes before any threshold would fire.

**Structural Poisoning Defense (Dual-Plane Verification)**
When an attacker compromises the monitoring infrastructure itself — injecting false telemetry — all legacy systems fail silently because they trust their data source. AutoNet-GNN maintains an independent physical-plane ground truth and continuously cross-references it against the logical graph. Inconsistent edges are pruned automatically; the system degrades gracefully rather than trusting corrupt data.

**Full Data Sovereignty**
All inference runs on-premises on an RTX 5060 / Ultra 9. No telemetry leaves the network perimeter. This ensures regulatory compliance for sensitive environments.

### Performance Comparison

| Dimension | Legacy (Reactive) | AutoNet-GNN (Predictive) |
|---|---|---|
| Detection Speed | Minutes – Hours | Sub-second (stream) |
| Alert Quality | 500+ raw alerts per event | 1 root-cause report |
| Threat Awareness | Known signatures only | Structural anomaly (unknown attacks) |
| Self-Healing | Manual remediation | Automated edge pruning + audit log |
| Data Sovereignty | Cloud SIEM (data egress) | On-prem RTX 5060 inference |

---

## How to Get Started

### Prerequisites

```bash
# Python environment
Python 3.10+
CUDA 12.x (for RTX 5060 inference)

# Core dependencies
pip install torch torch-geometric
pip install langchain openai chromadb
pip install netmiko pysnmp networkx plotly
pip install nflow-generator  # NetFlow simulation
```

### Recommended Infrastructure

- **Graph ML:** PyTorch Geometric (PyG) — GraphSAGE / GAT models
- **Temporal Modeling:** LSTM or Temporal Graph Networks (TGN)
- **Agent Layer:** LangChain + custom RAG over network knowledge base
- **Network Simulation:** GNS3 or EVE-NG (50–200 node synthetic topologies)
- **Data Collection:** Netmiko (SSH config), pysnmp (SNMP), nflow-generator (NetFlow)
- **Visualization:** NetworkX + Plotly / Gephi for live topology rendering

### Phased Build Roadmap

| Phase | Milestone | Stack | Difficulty |
|---|---|---|---|
| **MVP (Hackathon)** | GNN Config-Drift Detector | PyG + Netmiko + NetworkX | Easy — strongest live demo |
| Phase 2 | Lateral Movement GNN | GraphSAGE + internal NetFlow | Medium |
| Phase 3 | Pulse-Wave DDoS Detection | LSTM + CAIDA dataset | Medium |
| Phase 4 | Dual-Plane Poisoning Defense | Agentic RAG + cross-plane diff | Hard |
| Phase 5 | Full AutoNet-GNN Integration | All above + RTX 5060 CUDA | Stretch / Production |

### Quick Start: MVP Demo (60 seconds)

1. **Spin up a simulated 50-node topology** using GNS3 or EVE-NG
2. **Start the telemetry collector** — SNMP polling + Netmiko config snapshot
3. **Build the baseline graph** — NetworkX + PyG node/edge feature encoding
4. **Run the GNN** — GraphSAGE spatial pass + LSTM temporal baseline
5. **Inject a config change** — e.g., ACL modification on a core router via Netmiko
6. **Observe AutoNet-GNN output:**
   - `T+0s` — Live graph, all nodes green
   - `T+10s` — Config injection applied
   - `T+15s` — Legacy monitor: nothing (below threshold)
   - `T+25s` — AutoNet-GNN: deviant node flagged red, root-cause chain shown
   - `T+40s` — Agent proposes plain-English repair command
   - `T+55s` — One-click remediation; network returns to green; audit log written

### Open Datasets for Training

- **CAIDA Telescope Data** — DDoS pattern training (free academic access)
- **DARPA CICIDS 2017/2018** — Labeled intrusion detection with lateral movement scenarios
- **RouteViews + RIPE RIS** — BGP announcement history for hijacking detection baseline
- **SNMP MIB Walk Captures** — Interface degradation data from any lab switch

---

## Where to Get Help

### Documentation & Issues

- **Architecture Reference:** See the system diagram at the top of this README for a full component map
- **Bug Reports & Feature Requests:** Open an issue in this repository with the label `bug` or `enhancement`
- **Design Discussions:** Use GitHub Discussions for architectural questions, roadmap input, and integration proposals

### Community

- Review the `docs/` folder for detailed component guides on each layer (Telemetry Collector, GNN Pipeline, LangChain Agent, XAI Explainer)
- For RAG knowledge base setup (CVE corpus, runbooks, historical incident logs), see `docs/rag-setup.md`
- For synthetic topology creation guides (GNS3 / EVE-NG), see `docs/lab-setup.md`

---

## Who Maintains and Contributes

AutoNet-GNN is developed as a research and engineering initiative focused on next-generation cognitive network defense.

### Core Maintainers

| Role | Responsibility |
|---|---|
| GNN / ML Lead | GraphSAGE, GAT, TGN model development and training pipeline |
| Agent / RAG Lead | LangChain agent, CVE knowledge base, Chain-of-Thought reasoning |
| Network Engineering Lead | Telemetry collection, Netmiko integration, topology simulation |
| XAI / UX Lead | Explainability layer, operator console, audit log generation |

### Contributing

Priority contribution areas:
- Additional GNN model architectures (GAT variants, heterogeneous GNNs)
- New threat scenario scripts for the synthetic testbed
- Dataset adapters for additional telemetry sources
- XAI output improvements for non-technical operator audiences

---

*AutoNet-GNN • Cognitive Network Resilience • © AutoNet-GNN 2026*  
*RTX 5060 · On-Prem · Zero Data Egress*
