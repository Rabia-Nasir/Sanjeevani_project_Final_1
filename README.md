# Sanjeevani AI — Clinical Affective Intelligence & CanSupport Platform 🎗️

> **Universal Multimodal Emotion Recognition, Explainable Psycho-Oncology Reasoning, and Grounded CanSupport RAG System.**  
> Developed for **Sanjeevani – Life Beyond Cancer** NGO and cancer psycho-oncology support.

---

## ⚡ Quick Start (Run Locally in 2 Minutes)

Anyone can clone and run this project immediately on their PC. All AI models (Whisper-Base, XLM-RoBERTa, audEERING Wav2Vec2) download and cache automatically on the first launch.

### Windows (1-Click)
1. Clone the repository:
   ```bash
   git clone https://github.com/Rabia-Nasir/Sanjeevani_project_Final.git
   cd Sanjeevani_project_Final
   ```
2. Double-click **`run.bat`** (or open PowerShell/CMD and run `run.bat`).
3. That's it! The script installs requirements and launches the platform automatically.

---

### Mac / Linux / Manual Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Rabia-Nasir/Sanjeevani_project_Final.git
   cd Sanjeevani_project_Final
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the unified platform:
   ```bash
   python run_sanjeevani_platform.py
   ```

---

## 🌐 Accessing the Applications

Once launched, two interfaces are active simultaneously:

| Application | Local URL | Description |
| :--- | :--- | :--- |
| **Patient Portal UI** | **`http://localhost:8000`** | Modern responsive portal with Live Mic Recording, File Drop, Affective Biomarkers & Triage |
| **Evaluation Dashboard** | **`http://localhost:7860`** | 6-Tab mentor evaluation interface (Full Conversation Checkup, Chatbot RAG, SOAP, Admin) |
| **Interactive API Docs** | **`http://localhost:8000/docs`** | Swagger OpenAPI documentation for all REST endpoints |

---

## 🧠 Multimodal Architecture Highlights
- **Continuous Multimodal Fusion**: Calibrated audio acoustic prosody (pitch, jitter, RMS, ZCR) fused with neural semantic distress embeddings:
  $$\text{EDS} = \text{clamp}(0.55 \cdot S_{\text{semantic}} + 0.45 \cdot S_{\text{acoustic}}, 1.0, 99.0)$$
- **Acoustic Biomarkers**: Sub-band spectral energy, tremor detection, voice shaking, and speech tempo.
- **Explainable Psycho-Oncology**: Grounded PANAS-20 affective scale with automated SOAP clinical progress notes.
- **CanSupport RAG Engine**: Grounded retrieval from authentic Sanjeevani handbooks, programs, and regional hospital chapters.
