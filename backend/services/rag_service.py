"""
Sanjeevani AI - RAG & Chat Service
Reuses the verified SanjeevaniRAGEngine hybrid TF-IDF search, intent classification,
and grounded response synthesis.
"""

import os
import sys
import time
import uuid
from typing import List, Dict, Any

# Ensure workspace root and python code are in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from sanjeevani_rag_engine import get_rag_engine
from counsellor_dashboard import get_counsellor_manager
from backend.schemas.rag_schemas import ChatRequest, ChatResponse, PassageCitation
from backend.core.config import settings
from backend.core.logging import logger, mask_text_summary

class RAGInferenceService:
    def __init__(self):
        logger.info("Initializing RAGInferenceService (reusing verified SanjeevaniRAGEngine)...")
        self.rag_engine = get_rag_engine()
        self.counsellor_mgr = get_counsellor_manager()
        self.version = "Sanjeevani-RAG-v2.1"

    def answer_chat(self, req: ChatRequest, request_id: str) -> ChatResponse:
        t0 = time.perf_counter()
        query = req.message.strip()
        session_id = req.session_id or f"sess_{request_id[:8]}"
        
        logger.info(
            f"Processing RAG chat query: {mask_text_summary(query)}",
            extra={"request_id": request_id}
        )

        # 1. Use verified SanjeevaniRAGEngine grounded answer pipeline
        rag_output = self.rag_engine.generate_grounded_answer(
            query=query,
            eds=req.current_eds or 20.0,
            api_key=getattr(req, "api_key", None)
        )

        reply = rag_output["answer"]
        lang = rag_output["language"]
        intent = rag_output["intent"]
        intent_conf = rag_output["confidence"]
        new_eds = rag_output["eds"]
        tier = rag_output["triage_tier"]
        is_escalated = rag_output["is_escalated"]
        passages = rag_output.get("passages", [])

        # 2. Build passage citations
        citations_list: List[PassageCitation] = []
        for p in passages:
            meta = p.get("metadata", {})
            citations_list.append(PassageCitation(
                title=p.get("title", "Sanjeevani Program Guide"),
                category=p.get("category", "Cancer Care"),
                relevance_score=float(p.get("relevance_score", 0.85)),
                source=meta.get("source", "Sanjeevani Knowledge Base"),
                program_id=p.get("id"),
                contact=meta.get("contact"),
                eligibility=meta.get("eligibility")
            ))

        # Simple string sources for frontend compatibility
        sources_list: List[str] = [
            p.get("metadata", {}).get("source", p.get("title", "Sanjeevani Knowledge Base"))
            for p in passages
        ]
        if not sources_list and rag_output.get("citations"):
            sources_list = list(rag_output["citations"])

        # 7. AI-drafted reply suggestion if escalated & live insert into Counsellor Dashboard Queue
        suggested_counsellor = None
        if is_escalated:
            try:
                esc_item = self.counsellor_mgr.add_escalation(
                    query=query,
                    eds=new_eds,
                    lang=lang,
                    city="National Tele-Desk",
                    hospital="Sanjeevani Care Desk"
                )
                suggested_counsellor = esc_item.get("suggested_reply")
                logger.info(f"Chat escalated to Counsellor Queue: {esc_item.get('id')} (EDS: {new_eds})", extra={"request_id": request_id})
            except Exception as e:
                logger.error(f"Failed to add chat escalation to queue: {e}", extra={"request_id": request_id})
                suggested_counsellor = (
                    f"Dear Patient/Caregiver, we acknowledge your inquiry regarding '{query[:60]}...'. "
                    f"Senior counsellor at Sanjeevani has been flagged (Urgency: {tier}). "
                    f"Please call our direct national helpline at +91 8691000800 for immediate one-on-one handholding."
                )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return ChatResponse(
            request_id=request_id,
            session_id=session_id,
            reply=reply,
            answer=reply,
            detected_language=lang,
            detected_intent=intent,
            intent_confidence=round(intent_conf, 2),
            running_eds=new_eds,
            turn_eds=new_eds,
            triage_tier=tier,
            is_escalated=is_escalated,
            citations=citations_list,
            sources=sources_list,
            suggested_counsellor_reply=suggested_counsellor,
            safety_disclaimer=settings.SAFETY_DISCLAIMER,
            processing_time_ms=elapsed_ms
        )

rag_service = RAGInferenceService()
