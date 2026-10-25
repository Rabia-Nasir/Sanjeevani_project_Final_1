# Multi-stage production container for Sanjeevani AI
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    HOST=0.0.0.0

# Install essential system audio processing libraries (libsndfile & ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency requirements
COPY requirements.txt /app/requirements.txt

# Install python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application codebase
COPY backend /app/backend
COPY "python code" "/app/python code"
COPY knowledge_base /app/knowledge_base
COPY "patient portal.html" /app/
COPY counsellor_dashboard.py /app/
COPY clinical_soap_generator.py /app/
COPY sanjeevani_rag_engine.py /app/
COPY run_sanjeevani_platform.py /app/

# Expose ports for FastAPI (8000) and Gradio Evaluation UI (7860)
EXPOSE 8000
EXPOSE 7860

# Add Healthcheck against FastAPI endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Default launch command: runs unified local development & serving orchestrator
CMD ["python", "run_sanjeevani_platform.py"]
