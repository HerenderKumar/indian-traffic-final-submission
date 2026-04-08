# training/train_stage5_reptile.py
import yaml
import torch
import copy
import os

from maps.registry import MapRegistry
from env.hierarchical.hierarchical_env import HierarchicalTrafficEnv
from env.gnn_policy import TrafficGATActor, TrafficGATCritic
from training.mappo_trainer import MAPPOTrainer

def main():
    print("=" * 60)
    print("Stage 5: Reptile Meta-Learning (Domain Randomization)")
    print("=" * 60)

    print("Loading config...")
    with open("training/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Load environments for all three cities
    cities = ["pune", "chennai", "bengaluru"]
    envs = {}
    print("Initializing environments for Meta-Learning...")
    for city in cities:
        map_config = MapRegistry.load(city)
        # Using task_hard to force the network to learn robust features
        envs[city] = HierarchicalTrafficEnv(map_config, task_id="task_hard")

    # 2. Initialize Master Meta-Weights (Phi)
    print("Initializing Master Meta-Weights (Phi)...")
    meta_actor = TrafficGATActor(node_feature_dim=config['gnn_node_feature_dim'])
    meta_critic = TrafficGATCritic(node_feature_dim=config['gnn_node_feature_dim'])

    meta_step_size = 0.1  # Epsilon
    meta_iterations = 1000   # Dry-run (Production will be ~1000)
    inner_steps = 2048     # Steps per city per iteration

    print(f"\nStarting Reptile Meta-Training for {meta_iterations} iterations...")

    for iteration in range(1, meta_iterations + 1):
        for city in cities:
            print(f"  [Meta-Iter {iteration}/{meta_iterations}] Sampling task: {city.capitalize()}")
            
            # A. Clone meta-weights to task-specific models (W)
            task_actor = copy.deepcopy(meta_actor)
            task_critic = copy.deepcopy(meta_critic)
            
            trainer = MAPPOTrainer(task_actor, task_critic, envs[city], config)
            
            # B. Train on specific task (Inner Loop)
            # Suppress standard printing to keep the terminal clean during meta-loop
            trainer.learn(total_timesteps=inner_steps)
            
            # C. Reptile Update: Phi = Phi + epsilon * (W - Phi)
            with torch.no_grad():
                for meta_param, task_param in zip(meta_actor.parameters(), task_actor.parameters()):
                    meta_param.data += meta_step_size * (task_param.data - meta_param.data)
                    
                for meta_param, task_param in zip(meta_critic.parameters(), task_critic.parameters()):
                    meta_param.data += meta_step_size * (task_param.data - meta_param.data)

    print("\nReptile Meta-Learning dry-run complete!")
    
    # 3. Save Final Meta-Weights
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        'actor': meta_actor.state_dict(), 
        'critic': meta_critic.state_dict()
    }, "checkpoints/stage5_meta_weights.pt")
    print("Master Meta-weights saved to checkpoints/stage5_meta_weights.pt")

if __name__ == "__main__":
    main()