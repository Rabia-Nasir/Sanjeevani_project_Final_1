"""
Sanjeevani AI - Triage & Counsellor Service
Thread-safe management of escalation queues, case resolutions, and multi-turn session triage.
Guarantees that concurrent requests cannot corrupt JSON datastores.
"""

import os
import sys
import time
import json
import uuid
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from counsellor_dashboard import get_counsellor_manager
from clinical_soap_generator import get_soap_manager
import Sanjeevani_PANAS20_V2_App as app

from backend.schemas.triage_schemas import (
    SessionTriageRequest,
    SessionTriageResponse,
    EvaluatedTurnOutput,
    EscalationCase,
    EscalationQueueResponse,
    ResolveCaseRequest,
    ResolveCaseResponse,
    LongitudinalSessionItem,
    LongitudinalHistoryResponse
)
from backend.core.config import settings
from backend.core.logging import logger

# Thread lock for concurrent file persistence operations
QUEUE_LOCK = threading.Lock()
HISTORY_LOCK = threading.Lock()

class TriageAndEscalationService:
    def __init__(self):
        logger.info("Initializing TriageAndEscalationService (reusing Counsellor & SOAP managers with thread safety)...")
        self.counsellor_mgr = get_counsellor_manager()
        self.soap_mgr = get_soap_manager()
        self.version = "Sanjeevani-Triage-v2.1"

    def evaluate_session(self, req: SessionTriageRequest, request_id: str) -> SessionTriageResponse:
        t0 = time.perf_counter()
        session_id = req.session_id or f"sess_{request_id[:8]}"
        
        # Convert DialogueTurnInput list to format expected by consultation evaluator
        dialogue_turns = []
        for turn in req.turns:
            dialogue_turns.append({
                "start": turn.timestamp_start,
                "end": turn.timestamp_end,
                "speaker": turn.speaker,
                "text": turn.text,
                "prosody": {
                    "f0_std_hz": turn.f0_std_hz or 10.0,
                    "rms_mean": turn.rms_mean or 0.04,
                    "zcr_mean": turn.zcr_mean or 0.04,
                    "jitter_proxy": turn.jitter_proxy or 0.015,
                    "f0_mean_hz": 140.0
                }
            })

        # Evaluate dialogue turns using proven consultation analyzer logic
        evaluated_turns_out: List[EvaluatedTurnOutput] = []
        session_distress_list = []
        peak_moment = None
        peak_distress = -1.0
        all_entities = {}

        for idx, turn in enumerate(dialogue_turns):
            txt = turn["text"]
            prosody = turn["prosody"]
            
            # Project text to PANAS
            text_panas = app.project_text_to_panas(txt) or {
                "scores": {k: 1.0 for k in app.ALL_PANAS_ITEMS},
                "pa_sum": 10.0,
                "na_sum": 10.0,
                "has_escalation": False
            }
            
            # Extract entities
            ents = app.extract_clinical_entities_spacy(txt)
            for k, v in ents.items():
                all_entities[k] = sorted(list(set(all_entities.get(k, []) + v)))

            # Turn-level fused payload
            fused_turn = {
                "scores": text_panas["scores"],
                "pa_sum": text_panas["pa_sum"],
                "na_sum": text_panas["na_sum"],
                "has_escalation": text_panas["has_escalation"],
                "biomarkers": prosody,
                "cross_attention": None,
                "eval_text": txt
            }
            turn_triage = app.compute_eds_triage(fused_turn)
            turn_eds = turn_triage["eds"]
            session_distress_list.append(turn_eds)

            if turn_eds > peak_distress:
                peak_distress = turn_eds
                peak_moment = {
                    "time_str": f"{int(turn['start']//60):02d}:{int(turn['start']%60):02d}",
                    "text": txt,
                    "distress": turn_eds
                }

            st, et = turn["start"], turn["end"]
            evaluated_turns_out.append(EvaluatedTurnOutput(
                turn_num=idx + 1,
                timestamp=f"{int(st//60):02d}:{int(st%60):02d} - {int(et//60):02d}:{int(et%60):02d}",
                text=txt,
                turn_eds=turn_eds,
                vocal_status=turn_triage["badge"],
                primary_emotion=turn_triage["primary_emo"][2],
                entities=ents
            ))

        avg_distress = float(sum(session_distress_list) / max(1, len(session_distress_list)))
        session_eds = round(0.60 * avg_distress + 0.40 * max(0.0, peak_distress), 1)

        # Determine triage level & urgency
        escalate = session_eds > 60.0
        escalation_case_id = None
        if session_eds > 80.0:
            level = "4"
            urgency = "🔴 Priority Review Required"
        elif session_eds > 60.0:
            level = "3"
            urgency = "🟠 Counsellor Attention Recommended"
        elif session_eds > 30.0:
            level = "2"
            urgency = "🟡 Support Resources Offered"
        else:
            level = "1"
            urgency = "🟢 Healthy Conversational Balance"

        # Generate counselor guidance plan
        counselor_plan = app.generate_session_counselor_plan(session_eds, peak_moment, all_entities)

        # Thread-safe persistent escalation queue write if escalated
        if escalate:
            with QUEUE_LOCK:
                case_id = f"ESC-{int(time.time()*1000)%10000:04d}"
                escalation_case_id = case_id
                esc_item = self.counsellor_mgr.add_escalation(
                    query=peak_moment["text"] if peak_moment else (req.turns[0].text if req.turns else "High distress detected"),
                    eds=session_eds,
                    lang=req.language or "English",
                    city=req.hospital.split()[0] if req.hospital else "National Desk",
                    hospital=req.hospital or "Sanjeevani Tele-Desk"
                )
                escalation_case_id = esc_item.get("id", case_id)

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return SessionTriageResponse(
            request_id=request_id,
            session_id=session_id,
            session_eds=session_eds,
            peak_distress=peak_distress,
            triage_level=level,
            urgency_category=urgency,
            escalate_to_counsellor=escalate,
            escalation_case_id=escalation_case_id,
            evaluated_turns=evaluated_turns_out,
            counselor_plan=counselor_plan,
            causal_reasoning_html=f"<div>Peak distress detected: {peak_distress}% at {peak_moment['time_str'] if peak_moment else '00:00'}</div>",
            processing_time_ms=elapsed_ms,
            safety_disclaimer=settings.SAFETY_DISCLAIMER
        )

    def get_escalation_queue(self) -> EscalationQueueResponse:
        with QUEUE_LOCK:
            # Force reload from disk
            self.counsellor_mgr._load_queue()
            cases_raw = self.counsellor_mgr.queue
            
            pending = sum(1 for c in cases_raw if c.get("status") == "PENDING")
            resolved = sum(1 for c in cases_raw if c.get("status") == "RESOLVED")
            
            cases_out: List[EscalationCase] = []
            for c in cases_raw:
                cases_out.append(EscalationCase(
                    id=c.get("id", "ESC-000"),
                    timestamp=c.get("timestamp", ""),
                    session_id=c.get("session_id", ""),
                    patient_name=c.get("patient_name", "Anonymous"),
                    city=c.get("city", "Mumbai"),
                    hospital=c.get("hospital", "TMH"),
                    patient_query=c.get("patient_query", ""),
                    language=c.get("language", "English"),
                    eds=float(c.get("eds", 75.0)),
                    urgency=c.get("urgency", "High"),
                    status=c.get("status", "PENDING"),
                    suggested_reply=c.get("suggested_reply", ""),
                    counsellor_action=c.get("counsellor_action", "")
                ))
                
            return EscalationQueueResponse(
                total_cases=len(cases_out),
                pending_count=pending,
                resolved_count=resolved,
                cases=cases_out
            )

    def resolve_escalation_case(self, req: ResolveCaseRequest) -> ResolveCaseResponse:
        with QUEUE_LOCK:
            success = self.counsellor_mgr.resolve_escalation(
                esc_id=req.case_id,
                final_reply=f"Resolved by {req.counsellor_name}: {req.resolution_notes}"
            )
            msg = f"Case {req.case_id} resolved by {req.counsellor_name}" if success else f"Case {req.case_id} not found"
            return ResolveCaseResponse(
                status="success" if success else "not_found",
                case_id=req.case_id,
                message=msg,
                resolved_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

    def get_longitudinal_history(self) -> LongitudinalHistoryResponse:
        with HISTORY_LOCK:
            self.soap_mgr._load_history()
            records = self.soap_mgr.history
            items: List[LongitudinalSessionItem] = []
            for r in records:
                items.append(LongitudinalSessionItem(
                    day=r.get("day", 1),
                    date=r.get("date", ""),
                    stage=r.get("stage", "Checkup"),
                    pa=float(r.get("pa", 20.0)),
                    na=float(r.get("na", 20.0)),
                    eds=float(r.get("eds", 50.0)),
                    primary_emotion=r.get("primary_emotion", "Neutral")
                ))
            
            trend = "Stabilizing & Positive Recovery" if len(items) > 1 and items[-1].eds < items[0].eds else "Active Monitoring"
            return LongitudinalHistoryResponse(
                patient_journey="Sanjeevani 30-Day Recovery & Rehabilitation Cohort",
                total_sessions=len(items),
                trajectory=items,
                recovery_trend=trend
            )

triage_service = TriageAndEscalationService()
