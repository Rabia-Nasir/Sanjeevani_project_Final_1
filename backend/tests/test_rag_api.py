"""
Sanjeevani AI - RAG & Chat Endpoint Integration Tests
Validates POST /v1/chat (English & Hindi queries, citations, intent detection) and GET /v1/programs.
"""

import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_programs_catalog_endpoint():
    print("\n[TEST] Testing GET /v1/programs...")
    response = client.get("/v1/programs")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["total_programs"] == 14, f"Expected 14 programs, got {data['total_programs']}"
    assert len(data["programs"]) == 14
    titles = [p["title"] for p in data["programs"]]
    assert any("CanSahyogi" in t for t in titles), f"CanSahyogi not found in {titles}"
    assert any("CanAhaar" in t for t in titles), f"CanAhaar not found in {titles}"
    assert any("Satori" in t for t in titles), f"Satori not found in {titles}"
    print(f"✓ GET /v1/programs PASSED (14 Sanjeevani programs verified: {[t.split(' - ')[0] for t in titles]})")

def test_chat_canahaar_query():
    print("\n[TEST] Testing POST /v1/chat (CanAhaar Nutrition Query in English)...")
    payload = {
        "message": "What is CanAhaar and how do underprivileged cancer patients receive nutritional kits?"
    }
    response = client.post("/v1/chat", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert "CanAhaar" in data["reply"] or "nutrition" in data["reply"].lower()
    assert data["detected_language"] in ["English", "Hinglish"]
    assert "running_eds" in data
    assert "citations" in data
    assert len(data["citations"]) > 0
    print(f"✓ POST /v1/chat CanAhaar query PASSED (Language: {data['detected_language']}, Citations: {len(data['citations'])})")

def test_chat_hindi_query():
    print("\n[TEST] Testing POST /v1/chat (Hindi query for Tata Memorial Hospital counselor)...")
    payload = {
        "message": "टाटा मेमोरियल अस्पताल मुंबई में संजीवनी काउंसलर से कैसे मिलें?"
    }
    response = client.post("/v1/chat", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["detected_language"] in ["Hindi", "Hinglish"]
    assert len(data["reply"]) > 20
    print(f"✓ POST /v1/chat Hindi query PASSED (Language: {data['detected_language']})")

def test_chat_high_distress_and_emergency():
    print("\n[TEST] Testing POST /v1/chat (High distress emotional query & emergency)...")
    payload1 = {
        "message": "I feel so hopeless and terrified, chemo side effects are unbearable and I want to give up."
    }
    res1 = client.post("/v1/chat", json=payload1)
    assert res1.status_code == 200, f"Expected 200, got {res1.status_code}: {res1.text}"
    data1 = res1.json()
    assert data1["running_eds"] >= 55.0, f"Expected elevated EDS >= 55, got {data1['running_eds']}"
    print(f"✓ High Distress Query PASSED (EDS: {data1['running_eds']}%, Tier: {data1['triage_tier']})")

    # Critical emergency trigger
    payload2 = {
        "message": "I cannot bear this pain anymore, please help me, I want to die."
    }
    res2 = client.post("/v1/chat", json=payload2)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["running_eds"] >= 80.0, f"Expected emergency EDS >= 80, got {data2['running_eds']}"
    assert data2["is_escalated"] is True
    print(f"✓ Critical Emergency Query PASSED (EDS: {data2['running_eds']}%, Escalated: {data2['is_escalated']})")

if __name__ == "__main__":
    print("=== RUNNING SANJEEVANI RAG API TEST SUITE ===")
    test_programs_catalog_endpoint()
    test_chat_canahaar_query()
    test_chat_hindi_query()
    test_chat_high_distress_and_emergency()
    print("\n=======================================================")
    print(" ALL 4 RAG API INTEGRATION TESTS PASSED SUCCESSFULLY! ")
    print("=======================================================\n")
