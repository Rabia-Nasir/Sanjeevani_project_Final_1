"""
Sanjeevani AI - Text Analysis Router
Endpoint: POST /v1/analyze/text
"""

import uuid
from fastapi import APIRouter, HTTPException, status, Request
from backend.schemas.text_schemas import TextAnalysisRequest, TextAnalysisResponse
from backend.services.text_service import text_service
from backend.core.logging import logger

router = APIRouter(prefix="/v1/analyze", tags=["Text Analysis & Triage"])

@router.post(
    "/text",
    response_model=TextAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze patient text query or dialogue turn",
    description=(
        "Performs bilingual PANAS-20 lexical projection, clinical entity extraction, "
        "Section 8 Emotional Distress Score (EDS) triage, and Emotion-LLaMA causal reasoning."
    )
)
def analyze_text_endpoint(
    payload: TextAnalysisRequest,
    request: Request
) -> TextAnalysisResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    try:
        response = text_service.analyze(payload, request_id)
        return response
    except ValueError as ve:
        logger.warning(f"Validation error in analyze_text: {ve}", extra={"request_id": request_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Internal error during text analysis: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while evaluating affective signals. Please try again."
        )
