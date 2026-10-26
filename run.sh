#!/usr/bin/env bash
# Sanjeevani AI - 1-Click Platform Launcher for macOS & Linux

set -e

echo "======================================================================"
echo "      Sanjeevani AI - Clinical Affective Intelligence Platform"
echo "======================================================================"
echo ""

# Determine python command
if command -v python3 &>/dev/null; then
    PY_CMD="python3"
elif command -v python &>/dev/null; then
    PY_CMD="python"
else
    echo "[ERROR] Python 3 is not installed or not in PATH."
    exit 1
fi

# Optional: activate virtual environment if present
if [ -d "venv" ]; then
    echo "[*] Activating virtual environment..."
    source venv/bin/activate
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "[*] Using active virtual environment: $VIRTUAL_ENV"
fi

# Check if dependencies are installed
$PY_CMD -c "import fastapi, uvicorn, gradio, librosa" &>/dev/null || {
    echo "[1/2] Installing required dependencies..."
    pip install -r requirements.txt
    echo ""
}

echo "[2/2] Starting Sanjeevani AI Platform..."
echo "  - Patient Portal:  http://localhost:8000/portal"
echo "  - REST API Docs:   http://localhost:8000/docs"
echo "  - Evaluation UI:   http://localhost:7860"
echo ""

export PYTHONIOENCODING=utf-8
$PY_CMD run_sanjeevani_platform.py