"""
Sanjeevani AI - RAG & Chat Schemas
Request and response contracts for POST /v1/chat and POST /v1/retrieve.
Supports both 'query' and 'message' input parameters and dual response fields
for seamless compatibility with all frontend variants.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator

class ChatRequest(BaseModel):
    message: Optional[str] = Field(default=None, description="Patient message or question")
    query: Optional[str] = Field(default=None, description="Alternative field for patient query")
    language: Optional[str] = Field(default="auto", description="Preferred language (auto, English, Hindi, Hinglish)")
    session_id: Optional[str] = Field(default=None, description="Optional multi-turn session ID")
    current_eds: Optional[float] = Field(default=20.0, ge=0.0, le=100.0, description="Previous running EDS score")
    acoustic_stress: Optional[float] = Field(default=0.0, ge=0.0, le=100.0, description="Acoustic stress signal from voice if available")
    api_key: Optional[str] = Field(default=None, description="Optional Grok / Groq / OpenAI API key for live generation")

    @model_validator(mode="before")
    @classmethod
    def reconcile_query_message(cls, data: Any) -> Any:
        if isinstance(data, dict):
            msg = data.get("message") or data.get("query") or ""
            if not str(msg).strip():
                raise ValueError("Chat message or query cannot be empty or whitespace.")
            data["message"] = str(msg).strip()
            data["query"] = str(msg).strip()
        return data

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
    answer: str = Field(..., description="Alias for reply for frontend compatibility")
    detected_language: str = Field(..., description="English, Hindi, or Hinglish")
    detected_intent: str = Field(..., description="Classified intent (program_info, hospital_lookup, etc.)")
    intent_confidence: float
    running_eds: float = Field(..., ge=0.0, le=100.0, description="Updated Section 8 EDS score")
    turn_eds: float = Field(..., ge=0.0, le=100.0, description="Alias for running_eds for frontend compatibility")
    triage_tier: str = Field(..., description="Triage priority tier")
    is_escalated: bool = Field(..., description="True if escalated to human counsellor queue")
    citations: List[PassageCitation] = Field(default_factory=list, description="Verified knowledge-base citations")
    sources: List[str] = Field(default_factory=list, description="Simple list of source strings for frontend compatibility")
    suggested_counsellor_reply: Optional[str] = Field(None, description="AI-drafted reply suggestion for counselor")
    safety_disclaimer: str = Field(..., description="Non-clinical research and guidance disclaimer")
    processing_time_ms: float
