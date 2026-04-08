
class ZeroShotEmergencyGrader:
    BENGALURU_BASELINE = 85.0
    TARGET_WAIT        = 50.0
    CLEARANCE_TARGET   = 30.0
    COHERENCE_TARGET   = 0.75
    TASK_ID = "task_hard"

    def __init__(self):
        self.waits, self.emergency_times = [], []
        self.coherent_steps = self.total_steps = 0

    def on_step(self, reward, step: int):
        self.total_steps += 1
        self.waits.append(reward.global_avg_wait)
        if reward.emergency_cleared and reward.emergency_clearance_time is not None:
            self.emergency_times.append(reward.emergency_clearance_time)
        if all(r.wait_term > -0.9 for r in reward.per_intersection.values()):
            self.coherent_steps += 1

    def score(self) -> float:
        if not self.waits:
            return 0.0
        wait_score = max(0.0, min(1.0, (self.BENGALURU_BASELINE - sum(self.waits)/len(self.waits)) / (self.BENGALURU_BASELINE - self.TARGET_WAIT)))
        emergency_score = (sum(1 for t in self.emergency_times if t <= self.CLEARANCE_TARGET) / max(len(self.emergency_times), 1)) if self.emergency_times else 0.0
        coherence_score = min(1.0, (self.coherent_steps / max(self.total_steps, 1)) / self.COHERENCE_TARGET)
        return float(round(0.30 * wait_score + 0.40 * emergency_score + 0.30 * coherence_score, 4))

    def reset(self):
        self.waits = self.emergency_times = []
        self.coherent_steps = self.total_steps = 0