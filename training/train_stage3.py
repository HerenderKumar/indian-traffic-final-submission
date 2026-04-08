# training/train_stage3.py
import yaml
import torch

from maps.registry import MapRegistry
from env.multi_city_env import MultiCityTrafficEnv
from env.gnn_policy import TrafficGATActor, TrafficGATCritic
from training.mappo_trainer import MAPPOTrainer

def main():
    print("Loading config...")
    with open("training/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Loading City Configurations...")
    pune_cfg = MapRegistry.load("pune")
    chennai_cfg = MapRegistry.load("chennai")

    print("Initializing Multi-City Env (Pune 60%, Chennai 40%)...")
    # Stage 3 samples from Pune and Chennai
    env = MultiCityTrafficEnv(
        configs=[pune_cfg, chennai_cfg], 
        probabilities=[0.6, 0.4], 
        task_id="task_medium"
    )

    print("Initializing Graph Attention Networks (Actor & Critic)...")
    actor = TrafficGATActor(node_feature_dim=config['gnn_node_feature_dim'])
    critic = TrafficGATCritic(node_feature_dim=config['gnn_node_feature_dim'])
    
    print("Initializing MAPPO Trainer...")
    trainer = MAPPOTrainer(actor, critic, env, config)

    # Dry-run length
    test_steps = 200
    print(f"Starting Stage 3 Multi-City dry-run for {test_steps} steps...")
    
    trainer.learn(total_timesteps=test_steps)

    trainer.save("checkpoints/stage3_multicity.pt")
    print("Stage 3 Multi-City GAT dry-run complete and saved!")

if __name__ == "__main__":
    main()