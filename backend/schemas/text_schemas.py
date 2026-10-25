"""
Sanjeevani AI - Text Analysis Schemas
Request and response contracts for POST /v1/analyze/text.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, field_validator

class TextAnalysisRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Patient consultation transcript or typed query (English, Hindi, Hinglish)"
    )
    language: Optional[str] = Field(
        default="auto",
        description="Preferred language code ('auto', 'en', 'hi')"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session tracking ID for multi-turn dialogues"
    )

    @field_validator("text")
    @classmethod
    def validate_non_empty_whitespace(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Input text cannot be empty or purely whitespace.")
        return v.strip()

class PrimaryEmotion(BaseModel):
    key: str = Field(..., description="Machine-readable emotion category (happy, calm, anger, anxiety, sad, confident)")
    label: str = Field(..., description="Bilingual display label")
    score_pct: int = Field(..., ge=0, le=100, description="Normalized intensity percentage")
    color: str = Field(..., description="Hex color token")
    metric_name: str = Field(..., description="Clinical dimension name")
    action_guidance: str = Field(..., description="Empathetic counselor guidance")

class TriageResult(BaseModel):
    eds: float = Field(..., ge=0.0, le=100.0, description="Section 8 Emotional Distress Score (0-100)")
    level: str = Field(..., description="Triage priority level ('1', '2', '3', '4')")
    category: str = Field(..., description="Descriptive urgency tier")
    badge: str = Field(..., description="Visual emoji badge")
    color: str = Field(..., description="Hex indicator color")
    action: str = Field(..., description="Actionable triage recommendation")
    has_escalation: bool = Field(..., description="True if acute crisis phrase detected requiring human counselor")

class CausalReasoning(BaseModel):
    acoustic_evidence: str
    linguistic_evidence: str
    causal_attribution: str
    recommended_action: str

class TextAnalysisResponse(BaseModel):
    request_id: str = Field(..., description="Unique traceability identifier for this request")
    task: str = Field(default="lexical_affect_and_clinical_triage", description="Processing task name")
    model_version: str = Field(default="Sanjeevani-EmotionLLaMA-v2.1", description="Inference engine version")
    transcription_meta: Dict[str, Any] = Field(..., description="Transliteration and language metadata")
    panas20_scores: Dict[str, float] = Field(..., description="20-scale PANAS item ratings (1.0 - 5.0)")
    pa_sum: float = Field(..., ge=10.0, le=50.0, description="Positive Affect sum (10.0 - 50.0)")
    na_sum: float = Field(..., ge=10.0, le=50.0, description="Negative Affect sum (10.0 - 50.0)")
    primary_emotion: PrimaryEmotion = Field(..., description="Primary emotional state profile")
    triage: TriageResult = Field(..., description="Clinical triage and escalation assessment")
    clinical_entities: Dict[str, List[str]] = Field(..., description="Extracted clinical concepts by taxonomy")
    causal_reasoning: CausalReasoning = Field(..., description="Emotion-LLaMA structured reasoning breakdown")
    processing_time_ms: float = Field(..., description="Inference execution duration in milliseconds")
    warnings: List[str] = Field(default_factory=list, description="Any non-fatal runtime warnings")
    safety_disclaimer: str = Field(..., description="Non-diagnostic research and support disclaimer")
