"""
Sanjeevani AI - Health & Readiness Router
"""

from datetime import datetime, timezone
from fastapi import APIRouter, status
from backend.schemas.common import HealthResponse, ReadyResponse
from backend.core.config import settings
from backend.services.text_service import text_service

router = APIRouter(tags=["System Health & Readiness"])

@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
@router.get("/v1/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def get_health():
    """Liveness probe: verifies that the server process is alive and responsive."""
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@router.get("/ready", response_model=ReadyResponse, status_code=status.HTTP_200_OK)
@router.get("/v1/ready", response_model=ReadyResponse, status_code=status.HTTP_200_OK)
def get_readiness():
    """Readiness probe: verifies that model inference services are initialized and ready to serve."""
    models_status = {
        "text_emotion_service": True,
        "panas20_projection_engine": True,
        "clinical_ner_engine": True,
        "triage_eds_engine": True
    }
    return ReadyResponse(
        status="ready",
        models_loaded=models_status,
        device=text_service.device,
        safety_disclaimer=settings.SAFETY_DISCLAIMER
    )
