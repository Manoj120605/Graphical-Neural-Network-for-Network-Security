from .generate_topology import build_topology, assign_features, create_labels, to_pyg_data

# NOTE: data.ingest_docker requires Docker SDK and is NOT imported eagerly.
# Use explicit import when live ingestion is needed:
#   from data.ingest_docker import ingest
