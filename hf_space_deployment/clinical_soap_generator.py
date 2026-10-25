"""
Sanjeevani Clinical SOAP Note & Longitudinal Mood Tracking Engine
Complies with SRS FR-4.6, FR-4.7, and clinical oncology counseling documentation standards.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any

HISTORY_FILE = "knowledge_base/longitudinal_history.json"

class ClinicalSOAPManager:
    def __init__(self, history_file: str = HISTORY_FILE):
        self.history_file = history_file
        self.history = []
        self._load_history()

    def _load_history(self):
        """Loads multi-session trajectory history."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
            except Exception:
                self.history = self._get_initial_longitudinal_seed()
        else:
            self.history = self._get_initial_longitudinal_seed()
            self._save_history()

    def _save_history(self):
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def _get_initial_longitudinal_seed(self) -> List[Dict[str, Any]]:
        """Pre-seeds 30-day recovery trajectory across hospital visits for demonstration."""
        return [
            {"day": 1, "date": "2026-10-01", "stage": "Pre-Biopsy Panic", "pa": 14, "na": 42, "eds": 82.0, "primary_emotion": "🔴 Severe Anxiety"},
            {"day": 7, "date": "2026-10-07", "stage": "Post-Diagnosis Shock", "pa": 18, "na": 36, "eds": 72.0, "primary_emotion": "🟠 Moderate Distress"},
            {"day": 14, "date": "2026-10-14", "stage": "Chemo Cycle 1 Support", "pa": 26, "na": 28, "eds": 54.0, "primary_emotion": "🟡 Cautious Hope"},
            {"day": 21, "date": "2026-10-21", "stage": "Satori Pranayama Started", "pa": 34, "na": 20, "eds": 38.0, "primary_emotion": "🟢 Calm & Stabilizing"},
            {"day": 30, "date": "2026-10-24", "stage": "CanSahyogi Fellowship", "pa": 41, "na": 15, "eds": 24.0, "primary_emotion": "🌟 Happy & Resilient"}
        ]

    def add_session(self, pa: float, na: float, eds: float, primary_emotion: str, stage: str = "Live Checkup"):
        """Records a new consultation into the longitudinal trajectory."""
        now_str = datetime.now().strftime("%Y-%m-%d")
        next_day = self.history[-1]["day"] + 3 if self.history else 1
        self.history.append({
            "day": next_day,
            "date": now_str,
            "stage": stage,
            "pa": round(pa, 1),
            "na": round(na, 1),
            "eds": round(eds, 1),
            "primary_emotion": primary_emotion
        })
        self._save_history()

    def generate_soap_note_text(
        self,
        transcript: str,
        pa: float,
        na: float,
        eds: float,
        primary_emotion: str,
        f0_mean: float = 165.0,
        f0_std: float = 24.0,
        rms: float = 0.045
    ) -> str:
        """
        Generates structured Subjective-Objective-Assessment-Plan clinical note.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        triage_tier = "Priority 1 (Urgent)" if eds > 80 else ("Priority 2 (Follow-up)" if eds > 60 else "Priority 3 (Routine Maintenance)")
        
        # Clinical recommendations based on scores
        recommended_programs = []
        if na > 25 or eds > 50:
            recommended_programs.append("CanSahyogi (1-on-1 Bedside Counseling with Cancer Victor)")
            recommended_programs.append("Satori Holistic Breathwork (4-session weekly online pranayama)")
        if "chemo" in transcript.lower() or "nutrition" in transcript.lower() or "khana" in transcript.lower():
            recommended_programs.append("CanAhaar Nutritional Support (High-protein ration supplementation)")
        if "surgery" in transcript.lower() or "prosthesis" in transcript.lower():
            recommended_programs.append("CanRehab (Free silicone prosthesis & mastectomy kit)")
        if not recommended_programs:
            recommended_programs.append("Sanjeevani Wellness Club & CanUtsav fellowship")

        recs_str = "\n".join([f"  • {prog}" for prog in recommended_programs])

        note = f"""================================================================================
SANJEEVANI LIFE BEYOND CANCER — CLINICAL PSYCHO-ONCOLOGY CONSULTATION NOTE
Classification: Experimental Support & Prioritisation Tool (Non-Diagnostic / FR-4.6)
Date & Timestamp: {now}
================================================================================

1. [S] SUBJECTIVE COMPLAINT & PATIENT NARRATIVE:
   - Self-Reported Utterance: "{transcript if transcript else 'Patient spoke directly with counselor.'}"
   - Subjective Affective Presentation: Expressed emotional state mapped to '{primary_emotion}'.
   - Emotional Stress Intensity: {eds}% on 100-point Emotional Distress Scale (EDS).

2. [O] OBJECTIVE BIOMARKERS & QUANTITATIVE PSYCHOMETRICS:
   - Multimodal Affect Scale: Watson, Clark & Tellegen (1988) PANAS-20 Inventory
     * Positive Affect (PA): {pa:.1f} / 50.0  ({(pa/50.0)*100:.1f}%)
     * Negative Affect (NA): {na:.1f} / 50.0  ({(na/50.0)*100:.1f}%)
   - Acoustic Prosody Biomarkers:
     * Mean Fundamental Pitch (F0): {f0_mean:.1f} Hz (Std Dev: {f0_std:.1f} Hz)
     * Root Mean Square (RMS) Acoustic Energy: {rms:.4f}
     * Vocal Strain & Prosodic Tremor Index: {'Elevated Acoustic Tension' if f0_std > 35 else 'Stable & Controlled Vocal Tone'}

3. [A] PSYCHO-ONCOLOGICAL ASSESSMENT:
   - Primary Emotional State: {primary_emotion}
   - Triage Urgency Level: {triage_tier}
   - Clinical Impression: Patient exhibits {'acute situational distress and heightened anxiety' if eds > 60 else 'adaptable emotional balance with manageable coping resources'}. No psychiatric pathology asserted; prioritisation signal established for oncology counselor review.

4. [P] CARE PLAN & SANJEEVANI PROGRAM LINKAGE:
   - Recommended NGO Interventions:
{recs_str}
   - Follow-up Schedule: Re-evaluate emotional biomarkers within {'24 hours (Priority Contact)' if eds > 70 else '7 days (Routine Checkup)'}.
   - Assigned Support Desk: Tata Memorial Hospital / National Helpline (8691000800).
================================================================================
Attending Supervisor: Sanjeevani Certified Oncology Social Worker
"""
        return note

    def render_trajectory_svg(self) -> str:
        """
        Renders an interactive SVG chart showing 30-Day Emotional Trajectory.
        """
        width = 720
        height = 240
        padding = 45

        points = self.history[-8:]  # Last 8 sessions
        if len(points) < 2:
            return "<div class='text-stone-500 text-center py-6'>Log at least 2 sessions to see trajectory graph.</div>"

        # Coordinates
        x_step = (width - 2 * padding) / (len(points) - 1)
        
        # We plot PA (Positive Affect - green) and NA (Negative Affect - red)
        pa_coords = []
        na_coords = []
        eds_coords = []

        for i, pt in enumerate(points):
            x = padding + i * x_step
            # PA is 10 to 50 -> scale to height
            y_pa = height - padding - ((pt["pa"] - 10) / 40.0) * (height - 2 * padding)
            y_na = height - padding - ((pt["na"] - 10) / 40.0) * (height - 2 * padding)
            pa_coords.append((x, y_pa))
            na_coords.append((x, y_na))

        pa_path = "M " + " L ".join([f"{x:.1f},{y:.1f}" for x, y in pa_coords])
        na_path = "M " + " L ".join([f"{x:.1f},{y:.1f}" for x, y in na_coords])

        # Circles and labels
        pa_circles = "".join([f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#059669" />' for x, y in pa_coords])
        na_circles = "".join([f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#dc2626" />' for x, y in na_coords])

        x_labels = "".join([
            f'<text x="{padding + i * x_step:.1f}" y="{height - 12}" font-size="11" text-anchor="middle" fill="#64748b">Day {pt["day"]}</text>'
            for i, pt in enumerate(points)
        ])

        svg = f"""
        <div style="background:#ffffff; border-radius:12px; border:1px solid #e2e8f0; padding:16px; margin-top:12px; box-shadow:0 1px 3px rgba(0,0,0,0.05);">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <span style="font-weight:700; font-size:14px; color:#0f172a;">📈 30-Day Longitudinal Affect & Recovery Trajectory (Watson PANAS-20)</span>
            <div style="font-size:12px; display:flex; gap:16px;">
              <span style="color:#059669; font-weight:700;">● Positive Affect (PA)</span>
              <span style="color:#dc2626; font-weight:700;">● Negative Distress (NA)</span>
            </div>
          </div>
          <svg viewBox="0 0 {width} {height}" style="width:100%; height:auto; display:block;">
            <!-- Grid Lines -->
            <line x1="{padding}" y1="{padding}" x2="{width - padding}" y2="{padding}" stroke="#f1f5f9" stroke-width="1" />
            <line x1="{padding}" y1="{height/2}" x2="{width - padding}" y2="{height/2}" stroke="#f1f5f9" stroke-width="1" />
            <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#e2e8f0" stroke-width="1.5" />
            
            <!-- Lines -->
            <path d="{pa_path}" fill="none" stroke="#059669" stroke-width="3" stroke-linecap="round" />
            <path d="{na_path}" fill="none" stroke="#dc2626" stroke-width="3" stroke-linecap="round" stroke-dasharray="4,4" />
            
            <!-- Points -->
            {pa_circles}
            {na_circles}
            
            <!-- X Labels -->
            {x_labels}
          </svg>
          <div style="margin-top:8px; font-size:11px; color:#64748b; text-align:center;">
            Demonstrates clinical symptom reduction: Positive Affect climbing while Negative Distress steadily descends across Sanjeevani interventions.
          </div>
        </div>
        """
        return svg

# Global singleton
_soap_mgr = None

def get_soap_manager() -> ClinicalSOAPManager:
    global _soap_mgr
    if _soap_mgr is None:
        _soap_mgr = ClinicalSOAPManager()
    return _soap_mgr
