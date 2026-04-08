# env/multi_city_env.py
import numpy as np
from env.hierarchical.hierarchical_env import HierarchicalTrafficEnv

class MultiCityTrafficEnv:
    """
    Wraps HierarchicalTrafficEnv to sample randomly from multiple cities at reset.
    """
    def __init__(self, configs: list, probabilities: list, task_id: str = "task_medium"):
        assert len(configs) == len(probabilities), "Must provide a probability for each config."
        assert abs(sum(probabilities) - 1.0) < 1e-5, "Probabilities must sum to 1.0"
        
        self.configs = configs
        self.probabilities = probabilities
        self.task_id = task_id
        
        # Initialize the base environment with the first config
        self.current_env = HierarchicalTrafficEnv(self.configs[0], task_id=self.task_id)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        # Pick a city based on weights
        chosen_idx = np.random.choice(len(self.configs), p=self.probabilities)
        chosen_config = self.configs[chosen_idx]
        
        # If the city changed, swap the underlying map config
        if self.current_env._map_config.city != chosen_config.city:
            self.current_env = HierarchicalTrafficEnv(chosen_config, task_id=self.task_id)
            
        return self.current_env.reset(seed=seed, options=options)

    def step(self, actions):
        return self.current_env.step(actions)
        
    @property
    def _map_config(self):
        return self.current_env._map_config