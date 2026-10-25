"""
Sanjeevani AI - Automated Audio Inference Test Suite (Stage C)
Tests POST /v1/analyze/audio against benchmark recordings and failure edge cases.
Verifies temporary file cleanup, supported format validation, and output schema compliance.
"""

import os
import sys
import tempfile
import glob
import numpy as np
import soundfile as sf

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from fastapi.testclient import TestClient
from backend.main import app
import Sanjeevani_PANAS20_V2_App as gradio_engine

client = TestClient(app)

BENCHMARK_AUDIO_1 = os.path.join(WORKSPACE_DIR, "python code", "benchmark_audio_samples", "CASE_1_HINDI_ACUTE_PANIC.wav")
BENCHMARK_AUDIO_3 = os.path.join(WORKSPACE_DIR, "python code", "benchmark_audio_samples", "CASE_3_HINDI_REMISSION_JOY.wav")

def count_sanjeevani_temp_files():
    temp_dir = tempfile.gettempdir()
    return len(glob.glob(os.path.join(temp_dir, "sanjeevani_aud_*")))

def test_valid_audio_analysis():
    print("\n[TEST] 1. Testing POST /v1/analyze/audio with valid clinical benchmark audio...")
    assert os.path.exists(BENCHMARK_AUDIO_1), f"Benchmark file missing: {BENCHMARK_AUDIO_1}"
    
    initial_temps = count_sanjeevani_temp_files()
    
    with open(BENCHMARK_AUDIO_1, "rb") as f:
        response = client.post(
            "/v1/analyze/audio",
            files={"file": ("CASE_1_HINDI_ACUTE_PANIC.wav", f, "audio/wav")},
            data={"language": "hi"}
        )
        
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    
    # Schema assertions
    assert data["task"] == "multimodal_audio_prosody_and_triage"
    assert "audio_meta" in data
    assert data["audio_meta"]["sampling_rate"] == 16000
    assert data["audio_meta"]["channels"] == 1
    assert "biomarkers" in data
    assert data["biomarkers"]["f0_mean_hz"] > 0
    assert "triage" in data
    assert "primary_emotion" in data
    assert "safety_disclaimer" in data
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time" in response.headers
    
    # Temporary file cleanup assertion
    final_temps = count_sanjeevani_temp_files()
    assert final_temps == initial_temps, f"Temporary file leak detected! Initial: {initial_temps}, Final: {final_temps}"
    print(f"✓ Valid audio analysis PASSED (Duration: {data['audio_meta']['duration_sec']}s, Emotion: {data['primary_emotion']['label']}, Zero Temp Leaks)")

def test_equivalence_with_gradio_implementation():
    print("\n[TEST] 2. Testing numerical and categorical equivalence with existing Gradio pipeline...")
    assert os.path.exists(BENCHMARK_AUDIO_1)
    
    # 1. Direct call to existing Gradio app logic
    direct_fused = gradio_engine.fuse_multimodal_streams(None, BENCHMARK_AUDIO_1, "hi")
    direct_triage = gradio_engine.compute_eds_triage(direct_fused)
    
    # 2. Call via FastAPI endpoint
    with open(BENCHMARK_AUDIO_1, "rb") as f:
        api_resp = client.post(
            "/v1/analyze/audio",
            files={"file": ("CASE_1_HINDI_ACUTE_PANIC.wav", f, "audio/wav")},
            data={"language": "hi"}
        ).json()
        
    # Compare key metrics
    assert round(api_resp["triage"]["eds"], 1) == round(direct_triage["eds"], 1), "EDS mismatch with Gradio!"
    assert api_resp["primary_emotion"]["key"] == direct_triage["primary_emo"][0], "Primary emotion mismatch with Gradio!"
    assert round(api_resp["biomarkers"]["f0_mean_hz"], 1) == round(direct_fused["biomarkers"]["f0_mean_hz"], 1), "F0 mean mismatch!"
    assert round(api_resp["biomarkers"]["rms_mean"], 3) == round(direct_fused["biomarkers"]["rms_mean"], 3), "RMS mean mismatch!"
    print(f"✓ Equivalence verified: API EDS ({api_resp['triage']['eds']}%) exactly matches Gradio EDS ({direct_triage['eds']}%)!")

def test_unsupported_format_rejection():
    print("\n[TEST] 3. Testing format validation guard (rejection of unsupported file types)...")
    fake_payload = b"Sample text pretending to be an audio document."
    response = client.post(
        "/v1/analyze/audio",
        files={"file": ("patient_report.pdf", fake_payload, "application/pdf")}
    )
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    assert "Unsupported audio format" in response.json()["detail"]
    print(f"✓ Format validation guard PASSED (Rejected .pdf with HTTP 400: {response.json()['detail']})")

def test_empty_audio_rejection():
    print("\n[TEST] 4. Testing empty audio file rejection...")
    response = client.post(
        "/v1/analyze/audio",
        files={"file": ("empty.wav", b"", "audio/wav")}
    )
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    assert "empty" in response.json()["detail"].lower()
    print(f"✓ Empty audio rejection PASSED (Rejected 0-byte file with HTTP 400)")

def test_corrupt_audio_rejection():
    print("\n[TEST] 5. Testing corrupted audio payload handling and cleanup...")
    corrupt_bytes = b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00randomcorruptedbytes"
    initial_temps = count_sanjeevani_temp_files()
    
    response = client.post(
        "/v1/analyze/audio",
        files={"file": ("corrupt.wav", corrupt_bytes, "audio/wav")}
    )
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    
    final_temps = count_sanjeevani_temp_files()
    assert final_temps == initial_temps, "Temporary file not cleaned up after corrupt audio failure!"
    print("✓ Corrupted audio handled safely with HTTP 400 and guaranteed temporary cleanup")

def test_too_short_audio_rejection():
    print("\n[TEST] 6. Testing duration floor guard (< 0.35s)...")
    sr = 16000
    short_samples = np.zeros(int(0.1 * sr), dtype=np.float32)  # 0.1s
    
    buf_path = os.path.join(tempfile.gettempdir(), "test_too_short.wav")
    sf.write(buf_path, short_samples, sr)
    try:
        with open(buf_path, "rb") as f:
            response = client.post(
                "/v1/analyze/audio",
                files={"file": ("test_too_short.wav", f, "audio/wav")}
            )
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        assert "too short" in response.json()["detail"].lower()
        print("✓ Too-short audio duration rejected with HTTP 400")
    finally:
        if os.path.exists(buf_path):
            os.remove(buf_path)

if __name__ == "__main__":
    print("=================================================================")
    print("SANJEEVANI AI - FASTAPI AUDIO INFERENCE TEST SUITE (STAGE C)")
    print("=================================================================")
    test_valid_audio_analysis()
    test_equivalence_with_gradio_implementation()
    test_unsupported_format_rejection()
    test_empty_audio_rejection()
    test_corrupt_audio_rejection()
    test_too_short_audio_rejection()
    print("\n=================================================================")
    print("ALL AUDIO INFERENCE & SAFETY TESTS EXECUTED AND PASSED CLEANLY!")
    print("=================================================================")
