"""
Sanjeevani AI - Unified Local Platform Launcher
Starts both the FastAPI REST engine & Patient Portal (Port 8000)
and the 6-Tab Gradio Mentor Evaluation Dashboard (Port 7860).
"""

import sys
import os
import time
import signal
import threading
import uvicorn

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure workspace root and python code are in sys.path
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

PYTHON_CODE_DIR = os.path.join(WORKSPACE_DIR, "python code")
if PYTHON_CODE_DIR not in sys.path:
    sys.path.insert(0, PYTHON_CODE_DIR)

from backend.main import app as fastapi_app
from backend.core.config import settings

def run_fastapi_server():
    """Runs FastAPI on port 8000 serving API endpoints and Patient Portal."""
    config = uvicorn.Config(
        fastapi_app,
        host="0.0.0.0",
        port=settings.PORT,
        log_level="info",
        access_log=False
    )
    server = uvicorn.Server(config)
    server.run()

def run_gradio_dashboard():
    """Runs 6-tab Gradio app on port 7860 for mentor demonstrations."""
    import Sanjeevani_PANAS20_V2_App as gradio_module
    _static_dir = os.path.join(PYTHON_CODE_DIR, "static")
    gradio_module.demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        prevent_thread_lock=True,
        allowed_paths=[_static_dir]
    )

def print_banner():
    banner = f"""
========================================================================================
             🌿 SANJEEVANI AI — MULTIMODAL AFFECTIVE COMPUTING PLATFORM 🌿
                      Life Beyond Cancer | CanSupport AI Ecosystem
========================================================================================

 [1] FastAPI REST Engine & OpenAPI Docs:
     URL: http://127.0.0.1:{settings.PORT}
     Documentation: http://127.0.0.1:{settings.PORT}/docs
     Health Endpoint: http://127.0.0.1:{settings.PORT}/health

 [2] Sanjeevani Patient & Counselor Portal (UI):
     URL: http://127.0.0.1:{settings.PORT}/portal
     Features: 14 Programs, Hospital Directory, Voice Affect, RAG Chat, Counselor Queue

 [3] Mentor Research & Evaluation Dashboard (Gradio 6-Tab):
     URL: http://127.0.0.1:7860
     Tabs: Consultation Analyzer, Emotion Evaluation, RAG Chat, Counselor Escalation,
           SOAP Documentation, Portal Explorer

 Access Key for Protected Counselor Endpoints:
     Header: X-Counsellor-Key: {settings.COUNSELLOR_AUTH_KEY}

 [Press Ctrl+C to stop all local services]
========================================================================================
"""
    print(banner)

def main():
    print("[INIT] Starting Sanjeevani AI local platform services...")
    
    # 1. Start Gradio dashboard in background daemon thread
    gradio_thread = threading.Thread(target=run_gradio_dashboard, daemon=True, name="GradioDashboardThread")
    gradio_thread.start()
    
    # Allow Gradio to bind port
    time.sleep(2.0)
    
    # 2. Print operational banner
    print_banner()

    # 3. Start FastAPI in main thread
    run_fastapi_server()

if __name__ == "__main__":
    main()
