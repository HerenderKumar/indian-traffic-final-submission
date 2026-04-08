# training/mappo_trainer.py
import os
import torch
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

from env.graph_builder import build_graph

class MAPPOTrainer:
    def __init__(self, actor, critic, env, config):
        self.actor = actor
        self.critic = critic
        self.env = env
        self.config = config
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)
        
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), 
            lr=float(config.get('learning_rate', 3e-4))
        )

    def learn(self, total_timesteps: int):
        obs_dict = self.env.reset()
        
        for step in range(1, total_timesteps + 1):
            # 1. Convert Dictionary Obs to PyG Graph
            graph = build_graph(self.env._map_config, obs_dict).to(self.device)
            
            # 2. Forward pass through GAT
            self.actor.eval()
            with torch.no_grad():
                phase_logits, duration_logits = self.actor(graph.x, graph.edge_index, graph.edge_attr)
                
                # Sample discrete actions from the logits
                phase_dist = Categorical(logits=phase_logits)
                duration_dist = Categorical(logits=duration_logits)
                
                phases = phase_dist.sample()
                durations = duration_dist.sample()
            
            # 3. Format actions for the environment
            action_dict = {}
            for i, jid in enumerate(self.env._map_config.junction_ids):
                action_dict[jid] = np.array([phases[i].item(), durations[i].item()])
            
            # 4. Step the environment
            next_obs_dict, reward_dict, done_dict, info_dict = self.env.step(action_dict)
            obs_dict = next_obs_dict
            
            # 5. PPO Update (Simplified for dry-run verification)
            # In full production, this happens after collecting a batch of rollouts.
            if step % 50 == 0:
                self._dry_run_update(graph)
                print(f"  [MAPPO] Step {step}/{total_timesteps} — Gradients updated successfully.")
                
            if done_dict.get("__all__", False):
                obs_dict = self.env.reset()

    def _dry_run_update(self, graph):
        """Verifies that gradients flow correctly backward through the GAT architecture."""
        self.actor.train()
        self.critic.train()
        self.optimizer.zero_grad()
        
        # Forward pass tracking gradients
        p_logits, d_logits = self.actor(graph.x, graph.edge_index, graph.edge_attr)
        value = self.critic(graph.x, graph.edge_index, graph.edge_attr)
        
        # Dummy loss to trigger backprop
        loss = p_logits.mean() + d_logits.mean() + value.mean()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)

    @classmethod
    def load(cls, path, actor, critic, env, config):
        checkpoint = torch.load(path)
        actor.load_state_dict(checkpoint['actor'])
        critic.load_state_dict(checkpoint['critic'])
        return cls(actor, critic, env, config)