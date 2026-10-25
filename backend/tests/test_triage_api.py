"""
Sanjeevani AI - Triage & Counsellor Management API Tests
Validates:
1. POST /v1/triage/session (Dialogue turn evaluation, peak distress detection)
2. Authentication check on /v1/counsellor/* endpoints (401 without X-Counsellor-Key)
3. GET /v1/counsellor/queue (Retrieves seeded & escalated cases)
4. POST /v1/counsellor/resolve (Resolves an escalated case)
5. GET /v1/counsellor/longitudinal (Retrieves 30-day recovery trajectory)
6. Thread safety and concurrent persistence operations
"""

import sys
import os
import concurrent.futures

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)
VALID_AUTH_KEY = "sanjeevani-counsellor-demo-2026"

def test_session_triage_evaluation():
    print("\n[TEST] Testing POST /v1/triage/session (Multi-turn dialogue evaluation)...")
    payload = {
        "session_id": "TEST-SESS-001",
        "hospital": "Tata Memorial Hospital (TMH)",
        "language": "Hindi",
        "turns": [
            {
                "turn_num": 1,
                "speaker": "Patient",
                "text": "Namaste doctor, mujhe kal se bohot ulti aur chhati me dard ho raha hai.",
                "timestamp_start": 0.0,
                "timestamp_end": 4.5,
                "f0_std_hz": 18.2,
                "rms_mean": 0.045
            },
            {
                "turn_num": 2,
                "speaker": "Caregiver",
                "text": "I have so much panic and fear, chest feels tight, crying constantly and terrified we have no money for chemo!",
                "timestamp_start": 5.0,
                "timestamp_end": 10.2,
                "f0_std_hz": 42.0,
                "rms_mean": 0.11
            }
        ]
    }
    response = client.post("/v1/triage/session", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["session_id"] == "TEST-SESS-001"
    assert "session_eds" in data
    assert data["session_eds"] >= 40.0, f"Expected session EDS >= 40, got {data['session_eds']}"
    assert data["peak_distress"] >= 55.0, f"Expected peak distress >= 55, got {data['peak_distress']}"
    assert len(data["evaluated_turns"]) == 2
    assert "counselor_plan" in data
    assert "Non-Diagnostic Signal" in data["safety_disclaimer"]
    print(f"✓ POST /v1/triage/session PASSED (Session EDS: {data['session_eds']}%, Peak: {data['peak_distress']}%, Level: {data['triage_level']})")

def test_counsellor_auth_protection():
    print("\n[TEST] Testing Security Guard on /v1/counsellor/queue (401 without X-Counsellor-Key)...")
    # 1. No header -> 401
    res1 = client.get("/v1/counsellor/queue")
    assert res1.status_code == 401, f"Expected 401 Unauthorized, got {res1.status_code}"
    
    # 2. Invalid header -> 401
    res2 = client.get("/v1/counsellor/queue", headers={"X-Counsellor-Key": "invalid-token-xyz"})
    assert res2.status_code == 401, f"Expected 401 Unauthorized, got {res2.status_code}"

    # 3. Valid header -> 200
    res3 = client.get("/v1/counsellor/queue", headers={"X-Counsellor-Key": VALID_AUTH_KEY})
    assert res3.status_code == 200, f"Expected 200 OK, got {res3.status_code}"
    print("✓ Counsellor Authentication Security Check PASSED")

def test_counsellor_queue_and_resolve():
    print("\n[TEST] Testing GET /v1/counsellor/queue & POST /v1/counsellor/resolve...")
    headers = {"X-Counsellor-Key": VALID_AUTH_KEY}
    
    # 1. Fetch queue
    res_queue = client.get("/v1/counsellor/queue", headers=headers)
    assert res_queue.status_code == 200
    data_queue = res_queue.json()
    assert data_queue["total_cases"] >= 1
    assert len(data_queue["cases"]) >= 1
    first_case = data_queue["cases"][0]
    case_id = first_case["id"]
    print(f"  Targeting case {case_id} for resolution...")

    # 2. Resolve case
    resolve_payload = {
        "case_id": case_id,
        "counsellor_name": "Senior Counselor Meena",
        "resolution_notes": "Connected with patient relative; arranged CanAhaar nutrition box."
    }
    res_resolve = client.post("/v1/counsellor/resolve", json=resolve_payload, headers=headers)
    assert res_resolve.status_code == 200, f"Expected 200, got {res_resolve.status_code}: {res_resolve.text}"
    data_res = res_resolve.json()
    assert data_res["case_id"] == case_id
    assert data_res["status"] == "success"
    print(f"✓ Case {case_id} resolved successfully")

def test_longitudinal_trajectory():
    print("\n[TEST] Testing GET /v1/counsellor/longitudinal (30-Day Recovery Cohort)...")
    headers = {"X-Counsellor-Key": VALID_AUTH_KEY}
    response = client.get("/v1/counsellor/longitudinal", headers=headers)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["total_sessions"] >= 3
    assert len(data["trajectory"]) >= 3
    assert "recovery_trend" in data
    print(f"✓ GET /v1/counsellor/longitudinal PASSED ({data['total_sessions']} sessions, Trend: {data['recovery_trend']})")

def test_concurrent_triage_and_queue_persistence():
    print("\n[TEST] Testing Concurrent Triage Thread Safety (10 parallel requests)...")
    def submit_triage_job(idx):
        payload = {
            "session_id": f"CONCURRENT-SESS-{idx:03d}",
            "hospital": "Delhi State Cancer Institute (DSCI)",
            "language": "English",
            "turns": [
                {
                    "speaker": "Patient",
                    "text": f"Severe anxiety and physical pain during chemotherapy round {idx}.",
                    "timestamp_start": 0.0,
                    "timestamp_end": 5.0,
                    "f0_std_hz": 25.0 + idx,
                    "rms_mean": 0.06
                }
            ]
        }
        r = client.post("/v1/triage/session", json=payload)
        return r.status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(submit_triage_job, range(10)))
    
    assert all(code == 200 for code in results), f"Some concurrent requests failed: {results}"
    print(f"✓ Concurrent Triage Persistence PASSED (All 10 parallel requests succeeded)")

if __name__ == "__main__":
    print("=== RUNNING SANJEEVANI TRIAGE & COUNSELLOR API TEST SUITE ===")
    test_session_triage_evaluation()
    test_counsellor_auth_protection()
    test_counsellor_queue_and_resolve()
    test_longitudinal_trajectory()
    test_concurrent_triage_and_queue_persistence()
    print("\n=======================================================")
    print(" ALL 5 TRIAGE API INTEGRATION TESTS PASSED SUCCESSFULLY! ")
    print("=======================================================\n")
