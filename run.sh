#!/bin/bash
echo "======================================================================"
echo "      Sanjeevani AI - Clinical Affective Intelligence Platform"
echo "======================================================================"
echo ""
echo "[1/2] Installing dependencies..."
pip install -r requirements.txt
echo ""
echo "[2/2] Starting Sanjeevani AI Platform..."
echo " - Patient Portal: http://localhost:8000"
echo " - Evaluation UI: http://localhost:7860"
echo ""
python run_sanjeevani_platform.py