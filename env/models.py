
from __future__ import annotations
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator

class IntersectionState(BaseModel):
    junction_id: str
    queue_by_arm_and_type: List[List[float]] = Field(
        description="8 arms × 6 vehicle types, normalised [0,1]. Missing arms = 0.0"
    )
    current_phase: int = Field(ge=0, le=7)
    time_in_phase_normalised: float = Field(ge=0.0, le=1.0)
    emergency_present: bool
    emergency_direction: Optional[str] = None
    emergency_distance_normalised: float = Field(ge=0.0, le=1.0)
    avg_wait_seconds: float = Field(ge=0.0)
    queue_pressure: float = Field(ge=0.0)
    neighbor_congestion: Dict[str, float]
    coordinator_priority: float = Field(
        ge=0.5, le=2.0,
        description="NetworkCoordinator priority weight for this junction"
    )
    fairness_index: float = Field(
        ge=0.0, le=1.0,
        description="Jain's Fairness Index for this junction (1.0 = perfectly fair)"
    )

class TrafficObservation(BaseModel):
    city: str
    task_id: str
    step: int
    hour_of_day: float = Field(ge=0.0, lt=24.0)
    is_rush_hour: bool
    weather: str
    festival_active: bool = Field(description="True if Indian festival demand active")
    festival_type_id: int = Field(ge=0, description="0=none, 1=Ganesh, 2=Diwali, 3=IPL...")
    intersections: Dict[str, IntersectionState]
    flat_obs: Dict[str, List[float]] = Field(
        description="Pre-flattened (70,) vector per agent (69 base + 1 priority)"
    )
    episode_elapsed_seconds: float

class IntersectionAction(BaseModel):
    junction_id: str
    phase_index: int = Field(ge=0, le=7)
    duration_bucket: int = Field(ge=0, le=3)

    @validator("phase_index")
    def valid_phase(cls, v):
        if v not in range(8):
            raise ValueError(f"phase_index must be 0–7, got {v}")
        return v

class TrafficAction(BaseModel):
    actions: Dict[str, IntersectionAction]
    reasoning: Optional[str] = None

class IntersectionReward(BaseModel):
    junction_id: str
    total: float
    wait_term: float
    queue_pressure_term: float
    throughput_term: float
    emission_term: float
    emergency_bonus: float
    spillback_penalty: float
    phase_skip_bonus: float
    ped_bonus: float
    fairness_term: float = Field(description="Jain's Fairness Index reward contribution")

class TrafficReward(BaseModel):
    step_total: float
    per_intersection: Dict[str, IntersectionReward]
    global_avg_wait: float
    global_throughput: float
    global_fairness_index: float = Field(description="Mean JFI across all junctions")
    emergency_cleared: bool
    emergency_clearance_time: Optional[float]

class EnvironmentState(BaseModel):
    city: str
    task_id: str
    step: int
    max_steps: int
    done: bool
    episode_reward_total: float
    current_observation: TrafficObservation
    task_config: Dict[str, Any]
    map_config_summary: Dict[str, Any]
    coordinator_priorities: Dict[str, float] = Field(
        description="Current NetworkCoordinator priority weights"
    )