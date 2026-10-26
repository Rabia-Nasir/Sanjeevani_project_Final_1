# Sanjeevani AI — Clinical Affective Intelligence & CanSupport Platform

> **Universal Multimodal Emotion Recognition, Explainable Psycho-Oncology Reasoning, and Grounded CanSupport RAG System.**  
> Developed for **Sanjeevani – Life Beyond Cancer** NGO and cancer psycho-oncology support.

---

## 🚀 Quick Start (Run Locally in 2 Minutes)

Anyone can clone and run this project immediately on their computer (Windows, macOS, or Linux). All AI models and acoustic feature extractors load automatically on launch.

### Windows (1-Click)
1. Clone the repository:
   ```cmd
   git clone https://github.com/Rabia-Nasir/Sanjeevani_project_Final.git
   cd Sanjeevani_project_Final
   ```
2. Double-click **`run.bat`** (or in Command Prompt / PowerShell run `.\run.bat`).
3. That's it! It automatically verifies dependencies, starts the platform, and opens the Patient Portal in your browser.

---

### Mac / Linux (1-Click)
1. Clone the repository:
   ```bash
   git clone https://github.com/Rabia-Nasir/Sanjeevani_project_Final.git
   cd Sanjeevani_project_Final
   ```
2. Run the shell script:
   ```bash
   chmod +x run.sh
   ./run.sh
   ```

---

### Manual Run (Any OS)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the unified platform (FastAPI on :8000 + Gradio on :7860)
python run_sanjeevani_platform.py
```

---

## 🌐 Accessing the Applications

Once launched, two interfaces are active simultaneously:

| Application | URL | Description |
| :--- | :--- | :--- |
| **Patient & Counselor Portal** | **[http://localhost:8000/portal](http://localhost:8000/portal)** | Modern responsive portal with Live Mic Recording, File Upload, Affective Biomarkers & Triage |
| **Mentor Evaluation Dashboard** | **[http://localhost:7860](http://localhost:7860)** | 6-Tab mentor evaluation interface (Full Conversation Checkup, Chatbot RAG, SOAP, Queue) |
| **Interactive API Documentation** | **[http://localhost:8000/docs](http://localhost:8000/docs)** | Swagger OpenAPI documentation for all REST endpoints |

---

## 🧠 Multimodal Architecture Highlights

- **Continuous Multimodal Fusion**: Calibrated audio acoustic prosody (pitch, jitter, RMS energy, ZCR) fused with neural semantic distress embeddings:
  $$\text{EDS} = \text{clamp}(0.55 \cdot S_{\text{semantic}} + 0.45 \cdot S_{\text{acoustic}}, 1.0, 99.0)$$
- **Acoustic Biomarkers**: Sub-band spectral energy, tremor detection, voice shaking, and speech tempo analysis.
- **Explainable Psycho-Oncology**: Grounded PANAS-20 affective scale with automated SOAP clinical progress notes.
- **CanSupport RAG Engine**: Grounded retrieval from authentic Sanjeevani handbooks, 14 wellness programs, and regional hospital chapters.
- **Counselor Escalation Queue**: Thread-safe case escalation with non-diagnostic clinical triage prioritization.

---

## 🧪 Running Automated Tests

Run the full automated test suite (30 unit & integration tests) covering all endpoints, audio processing, and triage logic:
```bash
pytest backend/tests/ -v
```

---

## 📁 Repository Structure

```text
├── backend/                  # Production FastAPI REST Engine
│   ├── core/                 # Settings, CORS, logging, security
│   ├── routers/              # Audio, Text, RAG, Triage, Health endpoints
│   ├── schemas/              # Pydantic v2 validation contracts
│   ├── services/             # Audio, text, RAG, and escalation services
│   └── tests/                # 30+ automated tests (100% pass)
├── knowledge_base/           # Sanjeevani authentic JSON & vector documents
├── python code/              # 6-Tab Gradio Dashboard & Benchmark samples
│   ├── Sanjeevani_PANAS20_V2_App.py
│   ├── benchmark_audio_samples/
│   └── static/               # PWA assets & benchmark audio files
├── patient portal.html       # Standalone accessible patient web app
├── run_sanjeevani_platform.py # Unified dual-server orchestrator
├── run.bat                   # 1-click Windows launcher
├── run.sh                    # 1-click Mac/Linux launcher
├── Dockerfile                # Production multi-stage Docker container
└── requirements.txt          # Production dependencies
```
