
class ReduceWaitGrader:
    BASELINE_WAIT = 72.0
    TARGET_WAIT   = 25.0
    TASK_ID = "task_easy"

    def __init__(self):
        self.episode_waits = []

    def on_step(self, reward, step: int):
        self.episode_waits.append(reward.global_avg_wait)

    def score(self) -> float:
        if not self.episode_waits:
            return 0.0
        achieved = sum(self.episode_waits) / len(self.episode_waits)
        raw = (self.BASELINE_WAIT - achieved) / (self.BASELINE_WAIT - self.TARGET_WAIT)
        return float(max(0.0, min(1.0, raw)))

    def reset(self):
        self.episode_waits = []