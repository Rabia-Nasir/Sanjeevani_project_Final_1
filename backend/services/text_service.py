"""
Sanjeevani AI - Text Inference Service
Reuses the proven PANAS-20 lexical projection, clinical NER, and Emotion-LLaMA causal reasoning.
Guarantees identical mathematical behavior to the Gradio application.
"""

import os
import sys
import time
from typing import Dict, Any, List

# Ensure 'python code' directory is available on PYTHONPATH for exact function reuse
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY_CODE_DIR = os.path.join(BASE_DIR, "python code")
if PY_CODE_DIR not in sys.path:
    sys.path.insert(0, PY_CODE_DIR)

# Import proven inference functions directly from existing codebase
try:
    import Sanjeevani_PANAS20_V2_App as app
except ImportError as err:
    raise RuntimeError(f"Failed to load core Sanjeevani ML engine: {err}")

from backend.schemas.text_schemas import (
    TextAnalysisRequest,
    TextAnalysisResponse,
    PrimaryEmotion,
    TriageResult,
    CausalReasoning
)
from backend.core.config import settings
from backend.core.logging import logger, mask_text_summary

class TextInferenceService:
    def __init__(self):
        logger.info("Initializing TextInferenceService (reusing validated PANAS-20 and Emotion-LLaMA pipelines)...")
        self.version = "Sanjeevani-EmotionLLaMA-v2.1"
        self.device = getattr(app, "device", "cpu")
        self.has_neural = getattr(app, "text_emotion_pipe", None) is not None

    def analyze(self, req: TextAnalysisRequest, request_id: str) -> TextAnalysisResponse:
        t0 = time.perf_counter()
        raw_text = req.text.strip()
        
        logger.info(
            f"Processing text analysis request: {mask_text_summary(raw_text)}",
            extra={"request_id": request_id}
        )
        
        # 1. Devanagari detection and Hinglish transliteration
        has_devanagari = any('\u0900' <= c <= '\u097f' for c in raw_text)
        hinglish_text = app.hindi_to_hinglish(raw_text) if has_devanagari else raw_text
        
        # 2. Extract clinical entities via domain taxonomy
        entities = app.extract_clinical_entities_spacy(raw_text)
        
        # 3. Project text to 20-scale PANAS ratings
        text_panas = app.project_text_to_panas(raw_text)
        if not text_panas:
            # Fallback for unexpected empty parse
            text_panas = {
                "scores": {k: 1.0 for k in app.ALL_PANAS_ITEMS},
                "pa_sum": 10.0,
                "na_sum": 10.0,
                "has_escalation": False,
                "neural_dist": {}
            }
            
        scores = text_panas["scores"]
        pa_sum = text_panas["pa_sum"]
        na_sum = text_panas["na_sum"]
        
        # 4. Construct multimodal fusion payload for text-only mode
        fused_payload = {
            "mode": "Text-Only (XLM-RoBERTa + Lexical Projections)",
            "scores": scores,
            "pa_sum": pa_sum,
            "na_sum": na_sum,
            "has_escalation": text_panas["has_escalation"],
            "eval_text": raw_text,
            "biomarkers": None,
            "cross_attention": None,
            "clinical_entities": entities
        }
        
        # 5. Compute Section 8 Emotional Distress Score (EDS) Triage
        triage_data = app.compute_eds_triage(fused_payload)
        
        # 6. Primary Emotional State Profile
        primary_emo_tuple = triage_data["primary_emo"]
        # Tuple format: (key, score_pct, label, color, metric_name, plain_action)
        primary_emotion_obj = PrimaryEmotion(
            key=primary_emo_tuple[0],
            score_pct=primary_emo_tuple[1],
            label=primary_emo_tuple[2],
            color=primary_emo_tuple[3],
            metric_name=primary_emo_tuple[4],
            action_guidance=primary_emo_tuple[5]
        )
        
        triage_obj = TriageResult(
            eds=triage_data["eds"],
            level=triage_data["level"],
            category=triage_data["category"],
            badge=triage_data["badge"],
            color=triage_data["color"],
            action=triage_data["action"],
            has_escalation=text_panas["has_escalation"]
        )
        
        # 7. Emotion-LLaMA Structured Causal Reasoning
        # Extract individual structured clues
        reasoning_html = app.build_emotion_llama_reasoning(
            text=raw_text,
            prosody=None,
            scores=scores,
            entities=entities,
            eds=triage_data["eds"],
            is_session=False
        )
        
        # Build clean structured reasoning dictionary
        causal_obj = CausalReasoning(
            acoustic_evidence="Text analysis stream: Acoustic prosody is in conversational baseline or not provided.",
            linguistic_evidence=f"Themes identified: {list(entities.keys()) if entities else 'Conversational dialogue'}.",
            causal_attribution=primary_emo_tuple[5],
            recommended_action=triage_data["action"]
        )
        
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        
        warnings: List[str] = []
        if not self.has_neural:
            warnings.append("Neural classifier operating in lightweight lexical projection mode.")
            
        return TextAnalysisResponse(
            request_id=request_id,
            task="lexical_affect_and_clinical_triage",
            model_version=self.version,
            transcription_meta={
                "has_devanagari": has_devanagari,
                "hinglish": hinglish_text,
                "input_char_count": len(raw_text),
                "input_word_count": len(raw_text.split())
            },
            panas20_scores=scores,
            pa_sum=pa_sum,
            na_sum=na_sum,
            primary_emotion=primary_emotion_obj,
            triage=triage_obj,
            clinical_entities=entities,
            causal_reasoning=causal_obj,
            processing_time_ms=elapsed_ms,
            warnings=warnings,
            safety_disclaimer=settings.SAFETY_DISCLAIMER
        )

# Global singleton instance loaded once at worker startup
text_service = TextInferenceService()
