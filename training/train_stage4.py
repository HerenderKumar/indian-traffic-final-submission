# training/train_stage4.py
import yaml
import os
import torch
import torch.optim as optim

from maps.registry import MapRegistry
from env.hierarchical.hierarchical_env import HierarchicalTrafficEnv
from env.gnn_policy import TrafficGATActor, TrafficGATCritic
from training.mappo_trainer import MAPPOTrainer

def main():
    print("=" * 60)
    print("Stage 4: Hierarchical RL Training (L1 Coordinator + L2 GAT)")
    print("=" * 60)

    print("Loading config...")
    with open("training/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Initializing Hierarchical Env (Bengaluru)...")
    map_config = MapRegistry.load("bengaluru")
    env = HierarchicalTrafficEnv(map_config, task_id="task_hard")

    print("Initializing L1 Coordinator Optimizer...")
    # The coordinator is built directly into our custom environment wrapper
    coord_optimizer = optim.Adam(env.coordinator.parameters(), lr=1e-4)

    print("Initializing L2 GAT Policy...")
    actor = TrafficGATActor(node_feature_dim=config['gnn_node_feature_dim'])
    critic = TrafficGATCritic(node_feature_dim=config['gnn_node_feature_dim'])
    
    print("Initializing MAPPO Trainer...")
    trainer = MAPPOTrainer(actor, critic, env, config)

    test_steps = 200
    print(f"Starting Stage 4 Hierarchical dry-run for {test_steps} steps...")
    
    # 1. Step the environment (which triggers the Coordinator every 10 steps)
    trainer.learn(total_timesteps=test_steps)

    # 2. Verify L1 Coordinator Gradients Flow
    # We do a targeted backward pass for the Coordinator to prove it can train.
    env.coordinator.train()
    coord_optimizer.zero_grad()
    
    # Mock global state: n_agents * 3 features (Bengaluru = 5 agents * 3 = 15)
    mock_state = torch.rand(1, map_config.n_agents * 3)
    priorities = env.coordinator(mock_state)
    
    # Dummy loss to trigger backprop
    loss = priorities.mean()
    loss.backward()
    coord_optimizer.step()
    
    print("  [HRL] L1 Coordinator gradient update successful.")

    # 3. Save Checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    trainer.save("checkpoints/stage4_l2_gat.pt")
    torch.save(env.coordinator.state_dict(), "checkpoints/stage4_l1_coordinator.pt")
    
    print("Stage 4 Hierarchical dry-run complete and saved!")

if __name__ == "__main__":
    main()