# 🏆 Sanjeevani AI — One-Page Pitch Document
### *The Great Indian AI Internship Challenge 2026 Submission*

---

| **Project Name** | **Sanjeevani AI — Clinical Affective Intelligence & CanSupport Platform** |
|:---|:---|
| **Gnani AI Technology** | **Gnani Prisma v3.0 STT API** (`api.vachana.ai/stt/v3`) |
| **Target Domain** | Healthcare / Cancer Psycho-Oncology & Mental Health Triage |
| **Partner NGO Context** | Developed for *Sanjeevani – Life Beyond Cancer* NGO |
| **Repository** | [GitHub: Sanjeevani_project_Final_1](https://github.com/Rabia-Nasir/Sanjeevani_project_Final_1) |

---

## 1. Problem Statement
In India, over **1.4 million new cancer cases** are diagnosed annually. Cancer patients endure severe psychological distress—panic, chemotherapy fatigue, and existential dread. However:
- **Severe Counselor Shortage:** Only a handful of psycho-oncologists exist per thousand patients.
- **Linguistic & Literacy Barriers:** Most patients in Tier-2 and Tier-3 towns express emotional distress in **Hindi, Hinglish, or regional dialects**, making text-based chatbots unusable.
- **Noisy Telephonic Conditions:** Patients reach out via noisy phone calls, OPD waiting rooms, or home settings where standard speech recognizers fail.
- **Clinical Documentation Burden:** Counselors spend up to 40% of their time writing clinical SOAP notes instead of patient care.

---

## 2. Solution: Voice-First Affective Intelligence with Gnani Prisma v3.0
**Sanjeevani AI** is an end-to-end clinical voice triage and psycho-oncology support platform that listens to patients, evaluates distress, and empowers counselors.

```
┌─────────────────────────────────┐
│   Cancer Patient Voice Intake   │  (Hindi / Hinglish / Regional Audio)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│     Gnani Prisma v3.0 STT       │  ⚡ 130ms Low-Latency Transcription
│   (api.vachana.ai/stt/v3)       │  🌐 Multilingual Code-Mix & Telephonic Resilience
└────────────────┬────────────────┘
                 │
         ┌───────┴────────────────┐
         ▼                        ▼
┌──────────────────┐    ┌──────────────────┐
│ Acoustic Prosody │    │ Semantic Distress│
│ (Tremor, Jitter, │    │ (PANAS-20 Scales │
│  Pitch, RMS)     │    │  Cancer RAG KB)  │
└────────┬─────────┘    └────────┬─────────┘
         └───────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ Continuous Multimodal Fusion    │  EDS = 0.55 · Sem + 0.45 · Ac
└────────────────┬────────────────┘
                 │
         ┌───────┴────────────────┐
         ▼                        ▼
┌──────────────────┐    ┌──────────────────┐
│ Patient Portal   │    │ Counselor Dash   │
│ Real-Time Triage │    │ Auto SOAP Notes  │
└──────────────────┘    └──────────────────┘
```

---

## 3. Gnani AI Integration Details
- **Flagship Model:** **Prisma v3.0** (`model="prisma-v3.0"`), Gnani AI's state-of-the-art multilingual speech-to-text engine.
- **Integration Layer:** Standalone drop-in module [`gnani_prisma_stt.py`](file:///c:/Users/User/OneDrive/Documents/GitHub/Sanjeevani_project_Final/gnani_prisma_stt.py).
- **Endpoint:** `POST https://api.vachana.ai/stt/v3` authenticated via `X-API-Key-ID`.
- **Key Features Leveraged:**
  1. **Automatic Language Detection (`language_code="indic-auto"`):** Seamlessly recognizes Hindi (`hi-IN`), Indian English (`en-IN`), and regional speech without pre-selection.
  2. **Code-Mixed Formatting (`format="codemix"`):** Accurately captures colloquial Hinglish spoken by oncology patients.
  3. **Ultra-Low Latency:** Verified response times of **~0.13 seconds** for instantaneous clinical feedback.

---

## 4. Key Differentiators & Target Award Alignments
1. **Regional Languages & Hinglish Voice:** First-class support for Hindi and code-mixed medical dialogue.
2. **Noisy Telephonic Audio Handling:** Tested against noisy acoustic conditions, combining vocal tremor biomarkers with Prisma's robust acoustic modeling.
3. **Real-World Healthcare Impact:** Directly addresses the emotional trauma of cancer patients for *Sanjeevani – Life Beyond Cancer* NGO.
4. **Clinical-Grade Outputs:** Automatically compiles formal clinical **SOAP Progress Notes** (Subjective, Objective, Assessment, Plan) and maintains a priority escalation queue.

---

## 5. Technology Stack
- **Speech-to-Text:** Gnani AI Prisma v3.0 STT
- **Acoustic Feature Extraction:** Librosa, continuous VAD, F0 pitch tracking, jitter proxy
- **Clinical Framework:** PANAS-20 Affective Scale, Emotion Distress Score (EDS)
- **Backend:** FastAPI, Pydantic v2, Uvicorn (REST API)
- **Frontend & Dashboards:** HTML5/CSS3 Patient Portal + Gradio Multi-Tab Counselor Workspace
- **RAG Engine:** Grounded retrieval over authentic Sanjeevani cancer handbooks
