# tests/test_map_abstraction.py
import pytest
import numpy as np
from env.map_abstraction import MapAbstractionLayer
from env.graph_builder import build_graph
from maps.registry import MapRegistry

@pytest.fixture
def pune_config():
    return MapRegistry.load("pune")

def test_observation_shape_is_69(pune_config):
    mal = MapAbstractionLayer(pune_config)
    obs = mal.build_observation("dec_gym", {"N": 0.1, "S": 0.2, "E": 0.0, "W": 0.5}, "clear")
    assert obs.shape == (69,), f"Observation shape is {obs.shape}, expected (69,)"

def test_padding_logic(pune_config):
    mal = MapAbstractionLayer(pune_config)
    # Even if we change arms, queue length is always strictly 48
    queue_vec = mal.build_queue_vec("dec_gym")
    assert queue_vec.shape == (48,)

def test_graph_builder(pune_config):
    # Pass 70-float array simulating what the HierarchicalEnv outputs
    mock_obs = {jid: np.zeros(70, dtype=np.float32) for jid in pune_config.junction_ids}
    graph_data = build_graph(pune_config, mock_obs)
    
    assert graph_data.x.shape == (4, 70), "Node features should be (4, 70)"
    assert graph_data.edge_index.shape[0] == 2, "Edge index should have 2 rows"
    # Pune has 4 physical bidirectional roads = 8 directed edges
    assert graph_data.edge_attr.shape == (8, 4), "Edge attributes should be (8, 4)"