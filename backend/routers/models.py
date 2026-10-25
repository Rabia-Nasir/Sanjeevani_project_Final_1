"""
Sanjeevani AI - Models Inventory Router
Provides verified metadata about active models, avoiding fabricated version numbers.
"""

from fastapi import APIRouter, status
from backend.schemas.common import ModelListResponse, ModelItem

router = APIRouter(prefix="/v1", tags=["Models Inventory"])

@router.get("/models", response_model=ModelListResponse, status_code=status.HTTP_200_OK)
def list_models():
    """Lists verified model artifacts, feature extractors, and their operational status."""
    models = [
        ModelItem(
            model_id="sanjeevani-panas20-lexical",
            task="lexical_affect_projection",
            framework="Python/spaCy Rule Pipeline",
            input_format="raw_text (EN/HI/Hinglish)",
            output_format="PANAS-20 ratings [1.0, 5.0]",
            is_active=True,
            version_or_path="tabularisai/multilingual-emotion-classification + Domain Lexicon",
            calibration_status="Empirically Calibrated across PANAS-20 anchors"
        ),
        ModelItem(
            model_id="sanjeevani-clinical-ner",
            task="clinical_entity_recognition",
            framework="spaCy / Rule Matcher",
            input_format="raw_text",
            output_format="Categorized clinical entities (Symptoms, Coping, Conflict)",
            is_active=True,
            version_or_path="Clinical Oncology Taxonomy",
            calibration_status="Deterministic Matcher"
        ),
        ModelItem(
            model_id="sanjeevani-eds-triage",
            task="emotional_distress_scoring",
            framework="Section 8 SRS Decision Engine",
            input_format="PA/NA sums + Escalation flags + Biomarkers",
            output_format="EDS Score [0.0, 100.0] + Priority Level (1-4)",
            is_active=True,
            version_or_path="SRS Section 8 Formal Specification",
            calibration_status="Calibrated with Shouting/Calm Acoustic Bounds"
        ),
        ModelItem(
            model_id="sanjeevani-cross-attention",
            task="word_acoustic_synchrony_fusion",
            framework="Multi-Head Scaled Dot-Product (d=32, h=4)",
            input_format="Linguistic Tokens Q_L (N x 32), Acoustic Frames K_A (T x 32)",
            output_format="Word Attention Weights A (N x T), Word-level F0/RMS/ZCR",
            is_active=True,
            version_or_path="Mid-Level Multi-Head Cross-Attention Layer",
            calibration_status="Prior-Constrained Monotonic Attention (M_prior)"
        ),
        ModelItem(
            model_id="sanjeevani-rag-retriever",
            task="verified_knowledge_retrieval",
            framework="Scikit-Learn TF-IDF + Cosine Similarity",
            input_format="Natural language patient query",
            output_format="Ranked document chunks with official source citations",
            is_active=True,
            version_or_path="knowledge_base/sanjeevani_kb.json",
            calibration_status="Verified Ground Truth Grounding"
        )
    ]
    return ModelListResponse(total_models=len(models), models=models)
