# training/train_stage2.py
import yaml
import torch

from maps.registry import MapRegistry
from env.hierarchical.hierarchical_env import HierarchicalTrafficEnv
from env.gnn_policy import TrafficGATActor, TrafficGATCritic
from training.mappo_trainer import MAPPOTrainer

def main():
    print("Loading config...")
    with open("training/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Initializing Full 4-Node Multi-Agent Env (Stage 2)...")
    map_config = MapRegistry.load("pune")
    
    # FIX: Use HierarchicalTrafficEnv to ensure observations have 70 floats
    env = HierarchicalTrafficEnv(map_config, task_id="task_medium")

    print("Initializing Graph Attention Networks (Actor & Critic)...")
    actor = TrafficGATActor(node_feature_dim=config['gnn_node_feature_dim'])
    critic = TrafficGATCritic(node_feature_dim=config['gnn_node_feature_dim'])

    print(f"Device set to: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    print("Initializing MAPPO Trainer...")
    trainer = MAPPOTrainer(actor, critic, env, config)

    # Dry-run length
    test_steps = 200
    print(f"Starting MAPPO dry-run for {test_steps} steps...")
    
    trainer.learn(total_timesteps=test_steps)

    trainer.save("checkpoints/stage2_pune.pt")
    print("Stage 2 Multi-Agent GAT dry-run complete and saved!")

if __name__ == "__main__":
    main()