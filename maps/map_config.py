# maps/map_config.py
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class MapConfig:
    city: str
    cluster_name: str
    n_agents: int
    junction_ids: List[str]
    topology: Dict[str, Dict[str, str]]
    junction_arms: Dict[str, int]
    emergency_multipliers: Dict[str, float]
    emission_multipliers: Dict[str, float]
    ped_bonuses: Dict[str, float]
    net_file: str
    demand_files: Dict[str, str]
    road_capacity: Dict[str, int] = field(default_factory=dict)
    inter_junction_distance: Dict[str, int] = field(default_factory=dict)
    lane_id_map: Dict[str, List[str]] = field(default_factory=dict)
    festivals_file: str = ""   # path to city_festivals.yaml, empty = no festivals