"""
Sanjeevani AI - RAG & Chat Router
Endpoint: POST /v1/chat
"""

import uuid
from fastapi import APIRouter, HTTPException, status, Request
from backend.schemas.rag_schemas import ChatRequest, ChatResponse
from backend.services.rag_service import rag_service
from backend.core.logging import logger

router = APIRouter(prefix="/v1", tags=["RAG Conversational Chatbot"])

@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask Sanjeevani RAG chatbot a question",
    description="Grounded in 14 verified Sanjeevani programs with language detection and turn-level EDS triage."
)
@router.post(
    "/chat/message",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False
)
def chat_endpoint(payload: ChatRequest, request: Request) -> ChatResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    try:
        response = rag_service.answer_chat(payload, request_id)
        return response
    except ValueError as ve:
        logger.warning(f"Validation error in RAG chat: {ve}", extra={"request_id": request_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Internal error in RAG chat: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating the grounded response. Please try again."
        )

@router.get(
    "/programs",
    status_code=status.HTTP_200_OK,
    summary="List all 14 verified Sanjeevani programs",
    description="Returns verified oncology handholding, counseling, and nutrition programs."
)
def list_programs():
    docs = rag_service.rag_engine.documents
    programs = []
    for d in docs:
        if d.get("type") == "program":
            programs.append({
                "id": d.get("id"),
                "title": d.get("title"),
                "category": d.get("category"),
                "description": d.get("metadata", {}).get("description", ""),
                "eligibility": d.get("metadata", {}).get("eligibility", ""),
                "contact": d.get("metadata", {}).get("contact", "")
            })
    return {"total": len(programs), "total_programs": len(programs), "programs": programs}
