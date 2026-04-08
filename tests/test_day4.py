# tests/test_day5.py
from env.hierarchical.hierarchical_env import HierarchicalTrafficEnv
from maps.registry import MapRegistry

def test_hierarchical_env_100_steps():
    # Load Pune MapConfig
    config = MapRegistry.load("pune")
    env = HierarchicalTrafficEnv(config)
    
    # Reset Environment
    obs = env.reset()
    
    # Check that priority float was appended (69 -> 70 floats)
    for jid, agent_obs in obs.items():
        assert agent_obs.shape == (70,), f"Expected (70,), got {agent_obs.shape}"
        assert agent_obs[-1] == 1.0, "Initial priority should be exactly 1.0"
        
    # Run 100 steps
    for step in range(1, 101):
        # Sample random valid actions
        actions = {
            jid: env.action_space.spaces[jid].sample() 
            for jid in config.junction_ids
        }
        
        obs, rewards, done, info = env.step(actions)
        
        # Verify coordinator kicks in at step 10
        if step == 10:
            for jid, agent_obs in obs.items():
                priority = agent_obs[-1]
                assert 0.5 <= priority <= 2.0, "Priority bounded [0.5, 2.0]"
                # Since state is random, priority will likely have shifted from 1.0
                
    env.close()
    assert env._step_counter == 100, "Should have stepped 100 times"