"""
Sanjeevani AI - Audio Analysis Schemas
Contracts for POST /v1/analyze/audio endpoint.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

class AudioBiomarkers(BaseModel):
    rms_mean: float = Field(..., description="Root Mean Square energy (signal volume)")
    zcr_mean: float = Field(..., description="Zero-Crossing Rate (consonant frication / breathiness)")
    f0_mean_hz: float = Field(..., description="Fundamental vocal pitch mean in Hz")
    f0_std_hz: float = Field(..., description="Pitch variation standard deviation in Hz")
    jitter_proxy: float = Field(..., description="Micro-energy fluctuation (voice shakiness / tremor proxy)")

class AudioVAD(BaseModel):
    arousal: float = Field(..., ge=0.0, le=1.0, description="Nervous tension / autonomic activation (0.0 to 1.0)")
    dominance: float = Field(..., ge=0.0, le=1.0, description="Sense of agency vs helplessness (0.0 to 1.0)")
    valence: float = Field(..., ge=0.0, le=1.0, description="Affective valence (0.0 negative to 1.0 positive)")

class WordAlignment(BaseModel):
    word: str
    time_window: str
    f0_hz: float
    rms: float
    zcr: float
    attn_weight: float
    status: str
    tag_color: str
    bg_color: str

class CrossAttentionSynchrony(BaseModel):
    architecture: str = Field(default="Mid-Level Multi-Head Cross-Attention (d=32, h=4, Temporal Prior M_prior)")
    turning_point_word: Optional[str] = Field(None, description="Exact spoken word exhibiting peak vocal pitch/volume spike")
    diagnostic_insight: str = Field(..., description="Explainable turning-point synchrony insight")
    duration_sec: float
    word_alignments: List[WordAlignment]

class AudioTranscriptionMeta(BaseModel):
    transcript: str = Field(..., description="Recognized speech text")
    hinglish: str = Field(..., description="Romanized Hinglish transliteration")
    english_translation: str = Field(..., description="English translation if spoken in Hindi")
    detected_language: str

class AudioAnalysisResponse(BaseModel):
    request_id: str
    task: str = Field(default="multimodal_audio_prosody_and_triage")
    model_version: str = Field(default="Sanjeevani-EmotionLLaMA-v2.1")
    audio_meta: Dict[str, Any] = Field(..., description="Acoustic format metadata (duration, sampling rate, channels)")
    biomarkers: AudioBiomarkers = Field(..., description="Handcrafted acoustic features extracted via Librosa")
    vad: Optional[AudioVAD] = Field(None, description="Continuous dimensional affect (Arousal/Dominance/Valence)")
    transcription: AudioTranscriptionMeta
    panas20_scores: Dict[str, float]
    pa_sum: float
    na_sum: float
    primary_emotion: Dict[str, Any]
    triage: Dict[str, Any]
    cross_attention: Optional[CrossAttentionSynchrony] = None
    clinical_entities: Dict[str, List[str]]
    causal_reasoning: Dict[str, str]
    processing_time_ms: float
    warnings: List[str]
    safety_disclaimer: str
