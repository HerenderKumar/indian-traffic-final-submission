
import numpy as np
from maps.map_config import MapConfig

class MapAbstractionLayer:
    """
    Converts variable intersection geometries into a fixed 69-float observation.
    Standardizes 3-way, 4-way, and 5-way junctions into an 8-arm format.
    """
    def __init__(self, config: MapConfig):
        self.cfg = config

    def _get_active_festival(self, junction_id: str):
        # Stubbed for Day 3: Returns None. Will integrate YAML parsing later.
        return None

    def build_queue_vec(self, junction_id: str) -> np.ndarray:
        # 8 arms × 6 vehicle types = 48 floats
        # Real implementation will read traci.lane.getLastStepVehicleIDs
        return np.zeros(48, dtype=np.float32)

    def build_phase_vec(self, junction_id: str) -> np.ndarray:
        # 8-hot encoded phase + 1 time_in_phase_normalised = 9 floats
        vec = np.zeros(9, dtype=np.float32)
        vec[0] = 1.0  # mock phase 0 active
        return vec

    def build_time_vec(self) -> np.ndarray:
        # sin_time, cos_time, is_rush_hour = 3 floats
        return np.array([0.0, 1.0, 1.0], dtype=np.float32)

    def build_emergency_vec(self, junction_id: str) -> np.ndarray:
        # present, direction/3, distance/300 = 3 floats
        return np.zeros(3, dtype=np.float32)

    def build_neighbor_vec(self, neighbor_messages: dict) -> np.ndarray:
        # N, S, E, W congestion = 4 floats
        return np.array([
            neighbor_messages.get("N", 0.0),
            neighbor_messages.get("S", 0.0),
            neighbor_messages.get("E", 0.0),
            neighbor_messages.get("W", 0.0)
        ], dtype=np.float32)

    def build_weather_vec(self, weather: str) -> np.ndarray:
        # clear=0.0, rain/fog=1.0 = 1 float
        val = 1.0 if weather in ["rain", "fog"] else 0.0
        return np.array([val], dtype=np.float32)

    def build_festival_vec(self, junction_id: str) -> np.ndarray:
        # festival_type_id/5.0, demand_multiplier/4.0 = 2 floats
        if not self.cfg.festivals_file:
            return np.zeros(2, dtype=np.float32)
        active = self._get_active_festival(junction_id)
        if active is None:
            return np.zeros(2, dtype=np.float32)
        return np.array([
            active["festival_type_id"] / 5.0,
            active["demand_multiplier"] / 4.0,
        ], dtype=np.float32)

    def build_observation(self, junction_id: str, neighbor_messages: dict, weather: str) -> np.ndarray:
        """Assembles the full base vector and truncates to 69 floats as per spec."""
        obs = np.concatenate([
            self.build_queue_vec(junction_id),          # 48
            self.build_phase_vec(junction_id),          # 9
            self.build_time_vec(),                      # 3
            self.build_emergency_vec(junction_id),      # 3
            self.build_neighbor_vec(neighbor_messages), # 4
            self.build_weather_vec(weather),            # 1
            self.build_festival_vec(junction_id),       # 2
        ])
        # The sum is 70, but spec mandates returning 69 so HierarchicalEnv can append the priority.
        # We replace the final float (festival intensity) to make room for priority.
        obs_69 = obs[:69]
        assert obs_69.shape == (69,), f"Expected (69,), got {obs_69.shape}"
        return obs_69