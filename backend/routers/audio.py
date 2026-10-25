"""
Sanjeevani AI - Audio Analysis Router
Endpoint: POST /v1/analyze/audio
"""

import uuid
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Request
from backend.schemas.audio_schemas import AudioAnalysisResponse
from backend.services.audio_service import audio_service
from backend.core.logging import logger

router = APIRouter(prefix="/v1/analyze", tags=["Audio Analysis & Triage"])

@router.post(
    "/audio",
    response_model=AudioAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze patient voice recording or consultation audio",
    description=(
        "Accepts multi-format audio (.wav, .mp3, .ogg, .flac, .m4a), normalizes to 16 kHz mono, "
        "extracts Librosa acoustic biomarkers, audEERING continuous VAD, bilingual transcription, "
        "and computes Mid-Level Cross-Attention word-prosody synchrony."
    )
)
async def analyze_audio_endpoint(
    request: Request,
    file: UploadFile = File(..., description="Audio file in WAV, MP3, OGG, FLAC, or M4A format"),
    language: Optional[str] = Form("auto", description="Spoken language code ('auto', 'hi', 'en')")
) -> AudioAnalysisResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No audio file was uploaded. Please provide a valid audio file."
        )

    try:
        raw_bytes = await file.read()
    except Exception as read_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded audio file: {read_err}"
        )

    try:
        response = audio_service.process_and_analyze_audio(
            raw_bytes=raw_bytes,
            original_filename=file.filename,
            preferred_lang=language or "auto",
            request_id=request_id
        )
        return response
    except ValueError as ve:
        logger.warning(f"Audio validation failed: {ve}", extra={"request_id": request_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Internal error during audio analysis: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while evaluating voice acoustics. Please try again."
        )

@router.get(
    "/scenarios",
    status_code=status.HTTP_200_OK,
    summary="List preset clinical consultation scenarios",
    description="Returns preloaded multi-turn oncology consultation benchmark scenarios."
)
def list_consultation_scenarios():
    scenarios = audio_service.get_scenarios()
    return {"total": len(scenarios), "scenarios": scenarios}

@router.post(
    "/consultation",
    status_code=status.HTTP_200_OK,
    summary="Evaluate full consultation session (Audio file or Preset scenario)",
    description=(
        "Full conversation checkup: processes consultation audio or scenario, analyzes multi-turn "
        "dialogue, extracts vocal tone changes, computes longitudinal mood trajectory, "
        "identifies peak distress moments, and provides clinical counselor guidance plan."
    )
)
async def analyze_consultation_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None, description="Recorded session audio file"),
    scenario: Optional[str] = Form(None, description="Preloaded scenario title"),
    language: Optional[str] = Form("auto", description="Spoken language ('auto', 'hi', 'en')")
):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    raw_bytes = None
    filename = None
    
    if file and file.filename:
        try:
            raw_bytes = await file.read()
            filename = file.filename
        except Exception as read_err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read session audio file: {read_err}"
            )

    if not raw_bytes and not scenario:
        # Default to benchmark scenario if neither provided
        scenario = "Health Concern & Physical Exhaustion (Hindi / Hinglish)"

    try:
        result = audio_service.process_and_analyze_consultation(
            raw_bytes=raw_bytes,
            original_filename=filename,
            preferred_lang=language or "auto",
            sample_scenario=scenario,
            request_id=request_id
        )
        return result
    except ValueError as ve:
        logger.warning(f"Session validation failed: {ve}", extra={"request_id": request_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Internal error during session consultation analysis: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while evaluating the consultation session: {e}"
        )

