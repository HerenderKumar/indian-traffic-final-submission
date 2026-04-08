import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class FixedTimerPolicy:
    """
    A 'Dumb' baseline policy that mimics standard Indian traffic lights.
    It completely ignores observation data (queues, emergencies) and 
    just rotates phases on a strict, unyielding timer.
    """
    def __init__(self, cycle_duration=60, num_phases=4):
        self.cycle_duration = cycle_duration
        self.num_phases = num_phases
        
        
        self.junction_timers = {}
        self.junction_phases = {}

    def predict(self, state_request):
        """
        Receives the exact same JSON/State as the AI, but totally ignores 
        the queue lengths and emergency flags.
        """
        actions = []
        
        for intersection in state_request.intersections:
            j_id = intersection.junction_id
            
            # Initialize tracking for new junctions
            if j_id not in self.junction_timers:
                self.junction_timers[j_id] = 0
                self.junction_phases[j_id] = 0
                
            
            current_phase = self.junction_phases[j_id]
            
            actions.append({
                "junction_id": j_id,
                "next_phase": current_phase,
                "duration": self.cycle_duration
            })
            
            
            self.junction_timers[j_id] += self.cycle_duration
            if self.junction_timers[j_id] >= self.cycle_duration:
                self.junction_phases[j_id] = (self.junction_phases[j_id] + 1) % self.num_phases
                self.junction_timers[j_id] = 0

        return actions

# --- Quick Test ---
if __name__ == "__main__":
    from server.schemas import TrafficStateRequest, IntersectionState
    
    
    mock_state = TrafficStateRequest(
        city="bengaluru",
        intersections=[
            IntersectionState(
                junction_id="silk_board", 
                queue_lengths=[50, 10, 5, 2], 
                emergency_present=True, 
                current_phase=0
            )
        ]
    )
    
    dumb_timer = FixedTimerPolicy(cycle_duration=60)
    
    print("🚦 TESTING DUMB FIXED TIMER BASELINE 🚦")
    print(f"Scenario: Silk Board has 50 cars and an AMBULANCE waiting.")
    print("-" * 50)
    
    # Run a few cycles to show how blind it is
    for step in range(3):
        decisions = dumb_timer.predict(mock_state)
        action = decisions[0]
        print(f"Cycle {step + 1}: Junction {action['junction_id'].upper()} -> Assigned Phase P-{action['next_phase']} for {action['duration']} seconds.")
        print("   (Ambulance is still waiting...)")