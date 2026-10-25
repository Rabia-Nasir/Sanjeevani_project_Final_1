"""
Sanjeevani AI - Triage & Counsellor Schemas
Contracts for multi-turn session triage, escalation queue, case resolution, and longitudinal tracking.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class DialogueTurnInput(BaseModel):
    turn_num: Optional[int] = 1
    speaker: str = "Patient"
    text: str = Field(..., min_length=1)
    timestamp_start: float = 0.0
    timestamp_end: float = 5.0
    f0_std_hz: Optional[float] = 10.0
    rms_mean: Optional[float] = 0.04
    zcr_mean: Optional[float] = 0.04
    jitter_proxy: Optional[float] = 0.015

class SessionTriageRequest(BaseModel):
    session_id: Optional[str] = None
    patient_identifier: Optional[str] = "Anonymous"
    hospital: Optional[str] = "General / Tele-Support"
    language: Optional[str] = "auto"
    turns: List[DialogueTurnInput] = Field(..., min_length=1, description="Sequential dialogue turns")

class EvaluatedTurnOutput(BaseModel):
    turn_num: int
    timestamp: str
    text: str
    turn_eds: float
    vocal_status: str
    primary_emotion: str
    entities: Dict[str, List[str]]

class SessionTriageResponse(BaseModel):
    request_id: str
    session_id: str
    session_eds: float
    peak_distress: float
    triage_level: str
    urgency_category: str
    escalate_to_counsellor: bool
    escalation_case_id: Optional[str] = None
    evaluated_turns: List[EvaluatedTurnOutput]
    counselor_plan: str
    causal_reasoning_html: str
    processing_time_ms: float
    safety_disclaimer: str

class EscalationCase(BaseModel):
    id: str
    timestamp: str
    session_id: str
    patient_name: str
    city: str
    hospital: str
    patient_query: str
    language: str
    eds: float
    urgency: str
    status: str
    suggested_reply: str
    counsellor_action: str

class EscalationQueueResponse(BaseModel):
    total_cases: int
    pending_count: int
    resolved_count: int
    cases: List[EscalationCase]

class ResolveCaseRequest(BaseModel):
    case_id: str
    counsellor_name: str = "Authorized Counselor"
    resolution_notes: str = Field(..., min_length=3, description="Counselor clinical notes or direct reply sent")

class ResolveCaseResponse(BaseModel):
    status: str
    case_id: str
    message: str
    resolved_at: str

class LongitudinalSessionItem(BaseModel):
    day: int
    date: str
    stage: str
    pa: float
    na: float
    eds: float
    primary_emotion: str

class LongitudinalHistoryResponse(BaseModel):
    patient_journey: str = "Sanjeevani 30-Day Recovery & Rehabilitation Cohort"
    total_sessions: int
    trajectory: List[LongitudinalSessionItem]
    recovery_trend: str
