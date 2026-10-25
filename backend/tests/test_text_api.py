"""
Sanjeevani AI - Unit & API Integration Tests
Tests /health, /ready, /v1/models, and /v1/analyze/text using FastAPI TestClient.
"""

import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure workspace root is in sys.path
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_endpoint():
    print("\n[TEST] Testing GET /health...")
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data
    assert "X-Request-ID" in response.headers
    print("✓ GET /health PASSED")

def test_readiness_endpoint():
    print("\n[TEST] Testing GET /ready...")
    response = client.get("/ready")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "ready"
    assert data["models_loaded"]["text_emotion_service"] is True
    assert "Non-Diagnostic Signal" in data["safety_disclaimer"]
    print("✓ GET /ready PASSED")

def test_models_inventory():
    print("\n[TEST] Testing GET /v1/models...")
    response = client.get("/v1/models")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["total_models"] >= 4
    model_ids = [m["model_id"] for m in data["models"]]
    assert "sanjeevani-panas20-lexical" in model_ids
    assert "sanjeevani-eds-triage" in model_ids
    print(f"✓ GET /v1/models PASSED ({data['total_models']} models verified)")

def test_text_analysis_neutral():
    print("\n[TEST] Testing POST /v1/analyze/text (Neutral non-keyword text)...")
    payload = {"text": "Mera naam Rahul hai aur main kal subah market ja raha hu."}
    response = client.post("/v1/analyze/text", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["triage"]["eds"] <= 22.0, f"Expected calm EDS <= 22%, got {data['triage']['eds']}%"
    assert data["primary_emotion"]["key"] == "calm", f"Expected calm, got {data['primary_emotion']['key']}"
    assert "X-Request-ID" in response.headers
    print(f"✓ Neutral text analysis PASSED (EDS: {data['triage']['eds']}%, Emotion: {data['primary_emotion']['label']})")

def test_text_analysis_joy():
    print("\n[TEST] Testing POST /v1/analyze/text (Joyful Celebration)...")
    payload = {"text": "Mai bohot khush hu! Selected in leadership role and family is celebrating!"}
    response = client.post("/v1/analyze/text", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["triage"]["eds"] <= 12.0, f"Expected joy EDS <= 12%, got {data['triage']['eds']}%"
    assert data["primary_emotion"]["key"] == "happy"
    print(f"✓ Joy text analysis PASSED (EDS: {data['triage']['eds']}%, Emotion: {data['primary_emotion']['label']})")

def test_text_analysis_conflict():
    print("\n[TEST] Testing POST /v1/analyze/text (Interpersonal Conflict)...")
    payload = {"text": "You never listen to me! Why do you always ignore what I say during an argument?"}
    response = client.post("/v1/analyze/text", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["triage"]["eds"] >= 60.0, f"Expected conflict EDS >= 60% (Level 3), got {data['triage']['eds']}%"
    assert data["primary_emotion"]["key"] == "anger"
    print(f"✓ Conflict text analysis PASSED (EDS: {data['triage']['eds']}%, Emotion: {data['primary_emotion']['label']})")

def test_empty_text_validation():
    print("\n[TEST] Testing POST /v1/analyze/text with empty/whitespace input (Validation Guard)...")
    response = client.post("/v1/analyze/text", json={"text": "   "})
    assert response.status_code in [400, 422], f"Expected 400 or 422 for empty string, got {response.status_code}"
    print(f"✓ Empty text rejection PASSED (Returned HTTP {response.status_code})")

if __name__ == "__main__":
    print("=================================================================")
    print("SANJEEVANI AI - FASTAPI BACKEND TEST SUITE (STAGE A/B)")
    print("=================================================================")
    test_health_endpoint()
    test_readiness_endpoint()
    test_models_inventory()
    test_text_analysis_neutral()
    test_text_analysis_joy()
    test_text_analysis_conflict()
    test_empty_text_validation()
    print("\n=================================================================")
    print("ALL BACKEND ENDPOINT TESTS EXECUTED AND PASSED SUCCESSFULLY!")
    print("=================================================================")
