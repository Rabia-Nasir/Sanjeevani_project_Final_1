"""
Sanjeevani AI - Modular FastAPI Backend Engine
Production-grade REST API for Multimodal Affective Computing & CanSupport Triage
"""

import os
import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from backend.core.config import settings
from backend.core.logging import logger
from backend.routers import health, models, text, audio, rag, triage

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML models once per worker
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION} [{settings.ENVIRONMENT}]")
    yield
    # Shutdown: Clean up any active resources
    logger.info("Shutting down Sanjeevani AI backend engine.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Production-ready affective computing REST API for Sanjeevani – Life Beyond Cancer. "
        "Provides grounded PANAS-20 emotional profiling, Section 8 EDS triage, "
        "and explainable Emotion-LLaMA causal reasoning."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# 1. CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Process-Time"]
)

# 2. Request ID & Timing Middleware
@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = req_id
    t_start = time.perf_counter()
    
    response = await call_next(request)
    
    duration_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Process-Time"] = f"{duration_ms}ms"
    return response

# 3. Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"Unhandled server error: {exc}", extra={"request_id": req_id})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred while processing the affective evaluation.",
            "request_id": req_id
        }
    )

# 4. Include Routers
app.include_router(health.router)
app.include_router(models.router)
app.include_router(text.router)
app.include_router(audio.router)
app.include_router(rag.router)
app.include_router(triage.router)

# 5. Static Files & Patient Portal Mount
BASE_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_PROJECT_DIR, "python code", "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

PORTAL_HTML_PATH = os.path.join(BASE_PROJECT_DIR, "patient portal.html")
@app.get("/", response_class=FileResponse, tags=["Patient Portal"])
@app.get("/portal", response_class=FileResponse, tags=["Patient Portal"])
def get_patient_portal_page():
    """Serves the unified Sanjeevani clinical portal directly from FastAPI."""
    if os.path.exists(PORTAL_HTML_PATH):
        return FileResponse(PORTAL_HTML_PATH)
    return JSONResponse({"error": "Portal page not found on disk"}, status_code=404)

@app.get("/manifest.json", tags=["PWA"])
def get_pwa_manifest():
    """Serves PWA manifest.json from root for mobile installation."""
    manifest_file = os.path.join(STATIC_DIR, "manifest.json")
    if os.path.exists(manifest_file):
        return FileResponse(manifest_file, media_type="application/manifest+json")
    return JSONResponse({"error": "Manifest not found"}, status_code=404)

@app.get("/sw.js", tags=["PWA"])
def get_service_worker():
    """Serves PWA service worker with root scope."""
    sw_file = os.path.join(STATIC_DIR, "sw.js")
    if os.path.exists(sw_file):
        return FileResponse(sw_file, media_type="application/javascript")
    return JSONResponse({"error": "Service worker not found"}, status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
