# env/multi_agent_env.py
import gymnasium as gym
import numpy as np
from maps.map_config import MapConfig

class DummyReward:
    def __init__(self):
        self.total = 1.0

class MultiAgentTrafficEnv(gym.Env):
    def __init__(self, map_config: MapConfig, task_id: str = "task_medium"):
        self._map_config = map_config
        self.task_id = task_id
        self._last_obs = {}
        
        # Action space: MultiDiscrete([8 phases, 4 durations])
        self.action_space = gym.spaces.Dict({
            jid: gym.spaces.MultiDiscrete([8, 4]) for jid in map_config.junction_ids
        })
        
        # Base observation is 69 floats
        self.observation_space = gym.spaces.Dict({
            jid: gym.spaces.Box(low=0.0, high=1.0, shape=(69,), dtype=np.float32) 
            for jid in map_config.junction_ids
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._last_obs = {
            jid: np.zeros(69, dtype=np.float32) for jid in self._map_config.junction_ids
        }
        return self._last_obs

    def step(self, actions):
      
        obs_dict = {jid: np.random.rand(69).astype(np.float32) for jid in self._map_config.junction_ids}
        reward_dict = {jid: DummyReward() for jid in self._map_config.junction_ids}
        done_dict = {jid: False for jid in self._map_config.junction_ids}
        done_dict["__all__"] = False
        info_dict = {jid: {} for jid in self._map_config.junction_ids}
        
        self._last_obs = obs_dict
        return obs_dict, reward_dict, done_dict, info_dict

    def close(self):
        pass