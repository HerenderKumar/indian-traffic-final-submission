
class CooperativeFlowGrader:
    WEBSTERS_WAIT      = 50.0
    TARGET_WAIT        = 30.0
    MAX_SPILLBACK_RATE = 0.30
    TARGET_THROUGHPUT  = 0.85
    TARGET_FAIRNESS    = 0.85   
    TASK_ID = "task_medium"

    def __init__(self):
        self.waits, self.throughputs, self.fairness_scores = [], [], []
        self.spillback_events = 0
        self.total_steps = 0

    def on_step(self, reward, step: int):
        self.total_steps += 1
        self.waits.append(reward.global_avg_wait)
        self.throughputs.append(min(1.0, reward.global_throughput / 12.0))
        self.fairness_scores.append(reward.global_fairness_index)
        if any(r.spillback_penalty < -0.1 for r in reward.per_intersection.values()):
            self.spillback_events += 1

    def score(self) -> float:
        if not self.waits:
            return 0.0
        wait_score = max(0.0, min(1.0, (self.WEBSTERS_WAIT - sum(self.waits)/len(self.waits)) / (self.WEBSTERS_WAIT - self.TARGET_WAIT)))
        spillback_score = max(0.0, 1.0 - (self.spillback_events / max(self.total_steps, 1)) / self.MAX_SPILLBACK_RATE)
        throughput_score = min(1.0, (sum(self.throughputs)/len(self.throughputs)) / self.TARGET_THROUGHPUT)
        fairness_score = min(1.0, (sum(self.fairness_scores)/len(self.fairness_scores)) / self.TARGET_FAIRNESS)
        return float(round(0.35 * wait_score + 0.30 * spillback_score + 0.20 * throughput_score + 0.15 * fairness_score, 4))

    def reset(self):
        self.waits = self.throughputs = self.fairness_scores = []
        self.spillback_events = self.total_steps = 0