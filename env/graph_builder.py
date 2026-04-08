# env/graph_builder.py
import torch
from torch_geometric.data import Data
from maps.map_config import MapConfig

DIRECTION_TO_IDX = {"N": 0, "S": 1, "E": 2, "W": 3}

def _mock_get_flow(src_jid: str, dst_jid: str) -> float:
    """Mock SUMO edge flow for graph building without live simulation."""
    return 10.0

def build_graph(map_config: MapConfig, obs_per_junction: dict) -> Data:
    """
    Convert MapConfig topology + current state into a PyG Data object.
    Called every step before policy inference.
    """
    junction_ids = map_config.junction_ids
    jid_to_idx = {jid: i for i, jid in enumerate(junction_ids)}

    # Node features: (n_junctions, node_feature_dim)
    
    node_features = torch.stack([
        torch.tensor(obs_per_junction[jid], dtype=torch.float32)
        for jid in junction_ids
    ])

    edge_src, edge_dst, edge_attrs = [], [], []
    
    for src_jid, neighbors in map_config.topology.items():
        src_idx = jid_to_idx[src_jid]
        for direction, dst_jid in neighbors.items():
            dst_idx = jid_to_idx[dst_jid]
            
            # Edge features: [capacity_norm, flow_norm, distance_norm, direction_enc]
            capacity   = map_config.road_capacity.get(f"{src_jid}_{dst_jid}", 1000) / 2000.0
            flow       = _mock_get_flow(src_jid, dst_jid) / 1000.0
            distance   = map_config.inter_junction_distance.get(f"{src_jid}_{dst_jid}", 300) / 500.0
            direction_enc = DIRECTION_TO_IDX[direction] / 3.0
            
            edge_src.append(src_idx)
            edge_dst.append(dst_idx)
            edge_attrs.append([capacity, flow, distance, direction_enc])

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr  = torch.tensor(edge_attrs, dtype=torch.float32) if edge_attrs else torch.empty((0, 4))

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)