"""
Sanjeevani AI - Triage & Counsellor Management Router
Endpoints:
- POST /v1/triage/session (Multi-turn consultation checkup)
- GET  /v1/counsellor/queue (Protected escalation queue)
- POST /v1/counsellor/resolve (Protected case resolution)
- GET  /v1/counsellor/longitudinal (Protected longitudinal records)
"""

import os
import uuid
from typing import Optional
from fastapi import APIRouter, Header, HTTPException, status, Request
from backend.schemas.triage_schemas import (
    SessionTriageRequest,
    SessionTriageResponse,
    EscalationQueueResponse,
    ResolveCaseRequest,
    ResolveCaseResponse,
    LongitudinalHistoryResponse
)
from backend.services.triage_service import triage_service
from backend.core.logging import logger

router = APIRouter(prefix="/v1", tags=["Triage & Counsellor Operations"])

COUNSELLOR_AUTH_KEY = os.getenv("COUNSELLOR_AUTH_KEY", "sanjeevani-counsellor-demo-2026")

def verify_counsellor_access(x_counsellor_key: Optional[str] = Header(None)):
    """Guards protected counselor data against unauthenticated public access."""
    if not x_counsellor_key or x_counsellor_key != COUNSELLOR_AUTH_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access Denied: Valid X-Counsellor-Key header required to access patient escalation queue."
        )

@router.post(
    "/triage/session",
    response_model=SessionTriageResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate full multi-turn consultation session",
    description="Processes dialogue turns, identifies peak distress moment, generates counselor plan, and escalates if EDS > 60."
)
def evaluate_session_endpoint(payload: SessionTriageRequest, request: Request) -> SessionTriageResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    try:
        response = triage_service.evaluate_session(payload, request_id)
        return response
    except Exception as e:
        logger.error(f"Error evaluating session triage: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while evaluating the consultation session."
        )

@router.get(
    "/counsellor/queue",
    response_model=EscalationQueueResponse,
    status_code=status.HTTP_200_OK,
    summary="Get priority escalation queue (Counsellor only)",
    description="Requires X-Counsellor-Key header for authentication."
)
def get_escalation_queue_endpoint(
    x_counsellor_key: Optional[str] = Header(None)
) -> EscalationQueueResponse:
    verify_counsellor_access(x_counsellor_key)
    return triage_service.get_escalation_queue()

@router.post(
    "/counsellor/resolve",
    response_model=ResolveCaseResponse,
    status_code=status.HTTP_200_OK,
    summary="Resolve and close an escalation case (Counsellor only)",
    description="Requires X-Counsellor-Key header for authentication."
)
def resolve_case_endpoint(
    payload: ResolveCaseRequest,
    x_counsellor_key: Optional[str] = Header(None)
) -> ResolveCaseResponse:
    verify_counsellor_access(x_counsellor_key)
    return triage_service.resolve_escalation_case(payload)

@router.get(
    "/counsellor/longitudinal",
    response_model=LongitudinalHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get 30-day longitudinal recovery trajectory",
    description="Requires X-Counsellor-Key header for authentication."
)
def get_longitudinal_endpoint(
    x_counsellor_key: Optional[str] = Header(None)
) -> LongitudinalHistoryResponse:
    verify_counsellor_access(x_counsellor_key)
    return triage_service.get_longitudinal_history()
