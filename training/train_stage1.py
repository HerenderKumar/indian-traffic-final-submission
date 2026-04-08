# training/train_stage1.py
import os
import yaml
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from maps.registry import MapRegistry
from env.multi_agent_env import MultiAgentTrafficEnv

class SingleAgentPuneEnv(gym.Env):
    """
    Wraps the MultiAgent env to expose ONLY 'dec_gym' to Stable-Baselines3.
    This satisfies the Stage 1 'degenerate single-node' requirement.
    """
    def __init__(self):
        super().__init__()
        self.config = MapRegistry.load("pune")
        self.base_env = MultiAgentTrafficEnv(self.config, task_id="task_easy")
        self.target_agent = "dec_gym"
        
        # SB3 needs standard spaces, not Dict spaces
        self.observation_space = self.base_env.observation_space.spaces[self.target_agent]
        self.action_space = self.base_env.action_space.spaces[self.target_agent]

    def reset(self, seed=None, options=None):
        obs_dict = self.base_env.reset(seed=seed, options=options)
        return obs_dict[self.target_agent], {}

    def step(self, action):
        # Package the single action into a dict for the base env
        # Give dummy actions (all zeros) to the other 3 agents
        action_dict = {
            jid: np.array([0, 0]) if jid != self.target_agent else action
            for jid in self.config.junction_ids
        }
        
        obs_dict, reward_dict, done_dict, info_dict = self.base_env.step(action_dict)
        
        obs = obs_dict[self.target_agent]
        # In our mock env, reward_dict[jid] is a DummyReward object.
        # We extract the 'total' float.
        reward = float(reward_dict[self.target_agent].total) 
        
        # End episode after 200 steps for training
        done = done_dict.get("__all__", False)
        truncated = False 
        
        return obs, reward, done, truncated, info_dict[self.target_agent]

def main():
    print("Loading config...")
    with open("training/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Initializing Single Agent Env (Stage 1)...")
    env = SingleAgentPuneEnv()
    
    # Verify the wrapper complies with Gym API
    check_env(env, warn=True)
    print("Environment check passed!")

    print("Initializing PPO Agent...")
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=float(config["learning_rate"]),
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        ent_coef=config["entropy_coefficient"],
        verbose=1,
        tensorboard_log="./tensorboard_logs/stage1/"
    )

    # Deliverable Check: We won't run 500k steps right now, that takes hours!
    # We will run 2048 steps (1 rollout) to prove gradients flow and the env doesn't crash.
    test_steps = 2048
    print(f"Starting dry-run training for {test_steps} steps to verify pipeline...")
    
    model.learn(total_timesteps=test_steps)
    
    os.makedirs("checkpoints", exist_ok=True)
    model.save("checkpoints/stage1_pune_dryrun")
    print("Stage 1 dry-run complete and saved!")

if __name__ == "__main__":
    main()