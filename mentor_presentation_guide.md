# 🌿 Sanjeevani AI • Complete Mentor Presentation & Viva Master Guide
**Project Title:** Sanjeevani AI: Multimodal Affective Computing & Clinical Psycho-Oncology Triage Platform  
**Target Benchmark:** IEEE Transactions on Affective Computing (TAFFC) / CanSupport AI SRS Standards  
**Live Demo URLs:**
- **Patient & Counselor Portal:** [http://127.0.0.1:8000/portal](http://127.0.0.1:8000/portal)
- **FastAPI REST API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Mentor Research Evaluation Dashboard (Gradio 6-Tab):** [http://127.0.0.1:7860](http://127.0.0.1:7860)

---

## 1. Executive Summary & 2-Minute Elevator Pitch
> *"Sir / Ma'am, in oncology care, **over 60% of cancer patients and caregivers suffer from severe clinical distress, anxiety, and depression**, but hospitals face an acute shortage of psycho-oncologists. Furthermore, generic AI chatbots fail because they hallucinate medical advice, lack acoustic empathy, and cannot recognize acute oncological emergencies.*
> 
> *Our project, **Sanjeevani AI**, is a complete end-to-end multimodal affective computing and clinical triage platform developed for cancer care ecosystems. Rather than relying on simple text sentiment, our platform achieves three breakthroughs:*
> 1. *It decouples affect into **universal acoustic biomarkers** (pitch volatility, vocal strain, tremors, panting) and **multilingual semantic understanding** (Hindi, Hinglish, English) using a **Cross-Attention Multi-Head Fusion Engine**.*
> 2. *It implements a **strictly grounded, hallucination-free RAG engine** across 14 verified cancer support programs and 86 hospital desks, powered by bilingual Grok / Groq LLMs.*
> 3. *It features an **Acute Medical Emergency Protocol** (e.g., febrile neutropenia detection) and an **automated Counsellor Triage & Escalation Dashboard**, converting patient distress directly into real-time clinical queues and hospital-grade SOAP documentation.*
> 
> *Let me demonstrate how the complete system works live."*

---

## 2. Complete End-to-End Architecture ("How It Works")

```mermaid
flowchart TD
    subgraph Inputs ["1. Multimodal Inputs"]
        A1["Spoken Consultation Audio<br/>(WAV/MP3/Live Mic)"]
        A2["Text / Spoken Query<br/>(Hindi / Hinglish / English)"]
    end

    subgraph Acoustic ["2. Acoustic Affect Pipeline"]
        B1["Whisper ASR<br/>(Speech to Multilingual Text)"]
        B2["Librosa & audEERING VAD<br/>(F0 Pitch, RMS Volume, ZCR Tremor)"]
    end

    subgraph NLP ["3. Clinical Semantic NLP & RAG"]
        C1["PANAS-20 Emotion Projection<br/>(20 Psychometric Affect Dimensions)"]
        C2["Intent & Entity Extractor<br/>(Clinical Terms, Cities, Hospitals)"]
        C3["Grounded RAG Engine<br/>(Hybrid TF-IDF + Grok/Groq LLM)"]
    end

    subgraph Fusion ["4. Cross-Attention Synchrony"]
        D1["Mid-Level Cross-Attention Fusion<br/>(Acoustic Frames ⟷ Lexical Tokens)"]
        D2["Section 8 Distress Algorithm<br/>(EDS Score: 0 to 100%)"]
    end

    subgraph Safety ["5. Clinical Safety & Triage Engine"]
        E1{"Acute Emergency?<br/>(Chemo fever, bleeding, crisis)"}
        E2["🚨 Emergency Protocol<br/>(Immediate Oncologist / Casualty 112)"]
        E3{"EDS > 60%?<br/>(High Distress)"}
        E4["Priority Counsellor Escalation<br/>(Queue ESC-xxx + AI Reply Draft)"]
        E5["Empathetic Grounded Guidance<br/>(CanAhaar, CanSahyogi, Satori)"]
    end

    subgraph Out ["6. Clinical Outputs & Interfaces"]
        F1["Patient Web Portal<br/>(Interactive Chat & Visual Alert)"]
        F2["Counsellor Triage Dashboard<br/>(Queue Management & 1-Click Resolve)"]
        F3["Clinical SOAP Generator<br/>(Hospital Documentation & 30-Day Curve)"]
    end

    A1 --> B1
    A1 --> B2
    A2 --> C1
    B1 --> C1
    C1 --> D1
    B2 --> D1
    D1 --> D2
    A2 --> C3
    C2 --> C3
    D2 --> Safety
    C3 --> Safety
    E1 -- Yes --> E2
    E1 -- No --> E3
    E3 -- Yes --> E4
    E3 -- No --> E5
    E2 --> F1
    E4 --> F1
    E4 --> F2
    E5 --> F1
    D2 --> F3
```

---

## 3. Technology Stack: "Why We Used It" vs "Why Not Alternatives"

| Component / Technology | What We Used | Why We Used It (Clinical & Technical Justification) | Why Not Alternatives? |
| :--- | :--- | :--- | :--- |
| **Affect Representation** | **PANAS-20 + Section 8 EDS (0-100%)** | Validated psychometric scale measuring Positive Affect (PA: 10 items) and Negative Affect (NA: 10 items) plus emotional distress index. | Generic 3-class sentiment (*positive/negative/neutral*) is useless in healthcare; cancer patients experience fear, fatigue, and grief simultaneously. |
| **Multimodal Fusion** | **Mid-Level Cross-Attention Synchrony** | Dynamically aligns temporal acoustic frames (pitch spikes, tremors) with specific spoken words (e.g. pitch spike on *"pain"* or tremor on *"chemo"*). | Late decision averaging (e.g., $0.5 \times \text{audio} + 0.5 \times \text{text}$) loses word-level temporal context and fails on masked distress. |
| **Speech Recognition** | **OpenAI Whisper ASR + Roman Transliteration** | Robust zero-shot multilingual transcription for Hindi, Hinglish, and accented Indian English. | Standard Google Speech API fails on mixed Romanized Hinglish (*"bohot dar lag raha hai"*). |
| **Acoustic Biomarkers** | **Handcrafted Prosody ($F_0$, RMS, ZCR, Jitter)** | Zero-overhead, interpretable biomarkers that run in $<15\text{ ms}$ on standard CPU without requiring heavy GPU infrastructure. | End-to-end black-box deep models lack clinical explainability for oncologists and doctors. |
| **Knowledge Retrieval** | **Hybrid TF-IDF & Semantic RAG** | Offline-first, deterministic sub-15ms retrieval across 14 programs, 86 hospitals, and verified clinical protocols. | Cloud-only vector databases (Pinecone/Milvus) introduce latency, recurring cost, and privacy risks for hospital intranet deployments. |
| **Conversational LLM** | **Bilingual Grok (xAI) / Groq Cloud API** | Fast, empathetic conversational reasoning with automatic Hindi, Hinglish, and English prompt adaptation. | Generic ChatGPT lacks oncological guardrails and often suggests unverified dietary cures. |
| **Safety Guardrail** | **Deterministic Emergency Interceptor** | Hardcoded medical rules for red-flag symptoms: febrile neutropenia ($>100.4^\circ\text{F}$), active bleeding, intractable vomiting. | Never allow an LLM to "decide" whether fever during chemo is an emergency—clinical safety mandates rule-based deterministic override. |
| **Clinical Documentation** | **SOAP Progress Note Generator** | Formats patient affect into hospital-standard Subjective, Objective, Assessment, Plan notes ready for EHR export. | Unstructured chat transcripts are rejected by oncologists who only review structured SOAP notes. |

---

## 4. Step-by-Step Live Demonstration Script

### Step 1: Open Both Frontends in Your Browser
- Open Tab 1: **Patient Portal**: `http://127.0.0.1:8000/portal`
- Open Tab 2: **Mentor Evaluation Dashboard (Gradio)**: `http://127.0.0.1:7860`

---

### Step 2: Show the Core Problem Solved (Tab 2 of Gradio: Reviewer Sandbox)
1. Navigate to **Tab 2: Single Message or Voice Note Check**.
2. Click **Button 3: Masked Emotion ("I am totally fine")**:
   - **Explain to Mentor:** *"Look at this case, Sir. The patient says 'I am completely fine and normal'. A text sentiment analyzer gives this a Positive score. But our system analyzes their vocal tremor ($Jitter > 0.40$), overrides the deception, and detects **⚠️ MASKED DISTRESS** with an EDS score of 68%."*
3. Click **Button 1: Heated Conflict** or speak into the microphone loudly:
   - Point out: Pitch spike $> 220\text{ Hz}$ and volume $RMS > 0.08$ mapped to **Level 4 High Conflict**.
4. Click **Button 2: Work Burnout & Panic**:
   - Point out: Zero-crossing rate ($ZCR$) detecting panting and autonomic breathing agitation.

---

### Step 3: Show the RAG Chatbot & Oncological Safety (Patient Portal View 3)
1. In the Patient Portal, click on the **Ask Sanjeevani** tab.
2. Ask a normal oncology question:
   > *"What is CanAhaar and how can we get free food kits?"*
   - Show how it retrieves the verified CanAhaar eligibility protocol and source citation.
3. Now, test the **Bilingual Medical Emergency & Counselor Escalation**:
   > *"Mujhe chemo ke baad 102 bukhar ho gaya hai aur bohot dar lag raha hai"*
   - **Show the AI response:** It speaks warm Hindi, alerts them to **febrile neutropenia**, and demands immediate contact with their primary oncologist or casualty department.
   - **Highlight the NEW Interactive Badge:** A red glowing card appears right in the chat:
     > **🚨 Case Escalated to Counsellor Dashboard**  
     > *Critical Medical Emergency • EDS: 95.0%*  
     > `[ Open Counsellor Triage Queue → ]`
   - Notice the sidebar tab `Counsellor Triage` is now pulsing with a **`🔴 New`** badge!

---

### Step 4: Show the Counsellor Dashboard & 1-Click Resolution (Patient Portal View 4)
1. Click the red **`Open Counsellor Triage Queue →`** button inside the chat bubble.
2. The portal smoothly transitions to the **Counsellor Triage** view:
   - Show the newly escalated case right at the top of the queue with ID (e.g., `ESC-105`), urgency tier, timestamp, and the patient's exact quote.
   - Show the **AI-Drafted Response Suggestion** that saves the counselor 10 minutes of manual typing.
3. Click **"Mark Resolved"**:
   - The status updates immediately to **🟢 Resolved** with an auditable resolution timestamp!
4. *(Optional)* Switch to Gradio **Tab 4 (Counsellor Escalation & Admin Dashboard)**, click **"Refresh Escalation Queue"**, and show that the same case is synchronized in the clinical table with aggregate KPI metrics!

---

### Step 5: Show Multi-Turn Session Tracking & SOAP Notes (Gradio Tab 1 & Tab 5)
1. In Gradio **Tab 1: Full Conversation Checkup**:
   - Click **"3. Health Symptoms & Worry"** $\to$ Show the SVG Mood Trajectory Curve showing turn-by-turn distress progression.
2. In Gradio **Tab 5: Clinical SOAP Note & Mood Tracking**:
   - Click **"⚡ Generate Hospital SOAP Note"**:
   - Point out the **Subjective** (patient's spoken complaints), **Objective** (F0 pitch, RMS, acoustic biomarkers), **Assessment** (PANAS affect profile & triage level), and **Plan** (actionable clinical psycho-oncology recommendations).

---

## 5. Frequently Asked Mentor Questions & Confident Answers

#### Q1: "Where did you get your training data? Did you collect real hospital audio?"
> **Answer:** *"Sir, due to patient privacy laws (HIPAA and Indian Digital Personal Data Protection Act 2023), raw patient psychiatric recordings cannot be publicly open-sourced. Therefore, we followed the **IEEE TAFFC / NeurIPS 2024 Emotion-LLaMA research paradigm**:
> 1. We trained and calibrated our acoustic biomarker extractors on standardized benchmark speech databases (e.g., LibriSpeech, CREMA-D, RAVDESS).
> 2. We modeled text affect on the gold-standard **PANAS-20 clinical lexicon**.
> 3. We grounded the RAG knowledge base in official, published guidelines and program yearbooks from **Sanjeevani - Life Beyond Cancer**."*

#### Q2: "Why not just use an API like OpenAI GPT-4 for everything?"
> **Answer:** *"GPT-4 has three fatal drawbacks in clinical oncology triage:
> 1. **It cannot hear vocal acoustics:** It only receives text. If a patient says 'I am fine' while gasping or crying, GPT-4 is completely fooled.
> 2. **Hallucination Risk:** LLMs can hallucinate non-scientific medical advice or herbal remedies, which can be fatal for cancer patients. Our hybrid RAG restricts answers to verified documents.
> 3. **Latency & Cost:** A pure LLM call takes 2–4 seconds and costs per token. Our local deterministic pipeline processes acoustic affect in under 15ms with 0 API cost."*

#### Q3: "What is the difference between your EDS score and standard sentiment analysis?"
> **Answer:** *"Sentiment analysis is a 1-dimensional polarity (-1 to +1). It tells you whether words look negative, but it doesn't quantify psychiatric risk.
> Our **Emotional Distress Score (EDS, 0–100%)** is a clinical multi-factor score:
> - Base PANAS Negative Affect sum ($NA / 50 \times 100$).
> - Localized acoustic stress weighting (volume gradient, pitch volatility).
> - Red-flag clinical entity penalty (neutropenia, severe pain, suicidal crisis).
> This maps directly into 4 actionable triage levels: Normal, Emotional Support, Counsellor Attention, and Priority Emergency Review."*

#### Q4: "How does your system prevent medical malpractice liability?"
> **Answer:** *"We strictly enforce the Non-Functional Requirement **NFR-14 (Non-Diagnostic Disclaimer)**:
> 1. Every API response and UI screen includes a prominent medical disclaimer stating that Sanjeevani AI is a supportive psychosocial tool, not a doctor.
> 2. For acute emergencies (fever $>100.4^\circ\text{F}$ during chemo, hemorrhage, seizures), the system immediately halts guidance and redirects the patient to their treating oncologist and emergency hospital casualty."*

---

## 6. Checklist Before Your Review Tomorrow Morning
- [x] Both GitHub repositories fully synchronized and pushed.
- [x] `run.bat` automatically frees ports 8000/7860 and launches everything in 1 click.
- [x] Tested microphone input and verified audio permissions in browser.
- [x] Tested Hindi query: *"Mujhe chemo ke baad bukhar hai"* $\to$ Emergency alert + Counsellor Queue.
- [x] Tested Counsellor Triage "Mark Resolved" button.
- [x] Prepared browser bookmarks for `http://127.0.0.1:8000/portal` and `http://127.0.0.1:7860`.
