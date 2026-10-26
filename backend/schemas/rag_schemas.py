"""
Sanjeevani AI - RAG & Chat Schemas
Request and response contracts for POST /v1/chat and POST /v1/retrieve.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="Patient message or question")
    session_id: Optional[str] = Field(default=None, description="Optional multi-turn session ID")
    current_eds: Optional[float] = Field(default=20.0, ge=0.0, le=100.0, description="Previous running EDS score")
    acoustic_stress: Optional[float] = Field(default=0.0, ge=0.0, le=100.0, description="Acoustic stress signal from voice if available")
    api_key: Optional[str] = Field(default=None, description="Optional Grok / Groq / OpenAI API key for live generation")

    @field_validator("message")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Chat message cannot be empty or whitespace.")
        return v.strip()

class PassageCitation(BaseModel):
    title: str
    category: str
    relevance_score: float
    source: str
    program_id: Optional[str] = None
    contact: Optional[str] = None
    eligibility: Optional[str] = None

class ChatResponse(BaseModel):
    request_id: str
    session_id: str
    reply: str = Field(..., description="Grounded, empathetic answer for patient")
    detected_language: str = Field(..., description="English, Hindi, or Hinglish")
    detected_intent: str = Field(..., description="Classified intent (program_info, hospital_lookup, etc.)")
    intent_confidence: float
    running_eds: float = Field(..., ge=0.0, le=100.0, description="Updated Section 8 EDS score")
    triage_tier: str = Field(..., description="Triage priority tier")
    is_escalated: bool = Field(..., description="True if escalated to human counsellor queue")
    citations: List[PassageCitation] = Field(default_factory=list, description="Verified knowledge-base citations")
    suggested_counsellor_reply: Optional[str] = Field(None, description="AI-drafted reply suggestion for counselor")
    safety_disclaimer: str = Field(..., description="Non-clinical research and guidance disclaimer")
    processing_time_ms: float
