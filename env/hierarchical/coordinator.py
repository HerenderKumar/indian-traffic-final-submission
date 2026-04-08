# env/hierarchical/coordinator.py
import torch
import torch.nn as nn
import numpy as np

COORDINATOR_INTERVAL = 10    # runs every 10 simulation steps
PRIORITY_MIN = 0.5
PRIORITY_MAX = 2.0

class NetworkCoordinator(nn.Module):
    """
    High-level policy. Observes condensed state of ALL junctions.
    Outputs priority weight per junction agent every 10 steps.
    """
    def __init__(self, n_junctions: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_junctions * 3, hidden_dim),   # congestion + emergency_flag + phase per junction
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_junctions),
            nn.Sigmoid(),   # output in (0, 1) — scaled to [0.5, 2.0] below
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        raw = self.network(global_state)   
        return PRIORITY_MIN + raw * (PRIORITY_MAX - PRIORITY_MIN) 

    def get_priorities(self, obs_per_junction: dict) -> dict:
        features = []
        for jid in sorted(obs_per_junction.keys()):
            obs = obs_per_junction[jid]
            # Extract basic features from the 69-float base obs
            congestion  = float(np.mean(obs[:48])) if len(obs) >= 48 else 0.0         
            emerg_flag  = float(obs[60]) if len(obs) > 60 else 0.0                   
            phase_norm  = float(obs[49]) if len(obs) > 49 else 0.0  
            features.extend([congestion, emerg_flag, phase_norm])

        with torch.no_grad():
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            priorities = self.forward(input_tensor).squeeze(0).numpy()

        junction_ids = sorted(obs_per_junction.keys())
        return {jid: float(priorities[i]) for i, jid in enumerate(junction_ids)}