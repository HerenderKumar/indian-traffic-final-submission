
from env.multi_agent_env import MultiAgentTrafficEnv
from env.hierarchical.coordinator import NetworkCoordinator, COORDINATOR_INTERVAL
from maps.map_config import MapConfig
import numpy as np

class HierarchicalTrafficEnv(MultiAgentTrafficEnv):
    def __init__(self, map_config: MapConfig, task_id: str = "task_medium"):
        super().__init__(map_config, task_id)
        self.coordinator = NetworkCoordinator(n_junctions=map_config.n_agents)
        self.current_priorities = {jid: 1.0 for jid in map_config.junction_ids}
        self._step_counter = 0

    def step(self, actions):
        if self._step_counter % COORDINATOR_INTERVAL == 0:
            self.current_priorities = self.coordinator.get_priorities(self._last_obs)

        obs_dict, reward_dict, done_dict, info_dict = super().step(actions)

        # Append coordinator priority to each agent's observation (69 -> 70)
        for jid in obs_dict:
            priority = self.current_priorities.get(jid, 1.0)
            obs_dict[jid] = np.append(obs_dict[jid], priority)   

        # Scale rewards
        for jid in reward_dict:
            if hasattr(reward_dict[jid], 'total'):
                reward_dict[jid].total *= self.current_priorities.get(jid, 1.0)

        # Store base obs for the coordinator's next read
        self._last_obs = {jid: obs_dict[jid][:-1] for jid in obs_dict} 
        self._step_counter += 1
        
        return obs_dict, reward_dict, done_dict, info_dict

    def reset(self, seed=None, options=None):
        obs_dict = super().reset(seed=seed, options=options)
        self.current_priorities = {jid: 1.0 for jid in self._map_config.junction_ids}
        self._step_counter = 0
        self._last_obs = dict(obs_dict)
        
        for jid in obs_dict:
            obs_dict[jid] = np.append(obs_dict[jid], 1.0)
            
        return obs_dict