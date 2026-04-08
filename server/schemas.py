from pydantic import BaseModel
from typing import List

class IntersectionState(BaseModel):
    junction_id: str
    queue_lengths: List[int]        
    emergency_present: bool         
    current_phase: int              

class TrafficStateRequest(BaseModel):
    city: str                       
    intersections: List[IntersectionState]

class ActionResponse(BaseModel):
    junction_id: str
    next_phase: int                 
    duration: int                   

class TrafficActionResponse(BaseModel):
    actions: List[ActionResponse]