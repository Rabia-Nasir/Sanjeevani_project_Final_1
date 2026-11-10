"""
gnani_prisma_stt.py — Gnani Prisma v3.0 STT Wrapper for Sanjeevani AI
======================================================================
Standalone drop-in wrapper that calls Gnani AI's Prisma v3.0 Speech-to-Text
API (https://api.vachana.ai/stt/v3) and returns a transcript string.

This file does NOT modify any existing Sanjeevani code.
It can be imported anywhere in the project to replace or augment the
existing SpeechRecognition-based transcription.

Setup:
  1. Set GNANI_API_KEY in your environment or .env file
  2. pip install requests  (already in requirements.txt)
  3. Import and call transcribe_audio_bytes() or transcribe_audio_file()

Supported audio: WAV, MP3, OGG, FLAC, AAC, M4A (max ~30s per clip)
Supported languages: indic-auto (auto-detect), hi-IN, en-IN, kn-IN,
                     ta-IN, te-IN, ml-IN, bn-IN, mr-IN, gu-IN, pa-IN

Great Indian AI Internship Challenge — Gnani.ai
Project: Sanjeevani AI — Clinical Affective Intelligence for Cancer Support
"""

import os
import io
import logging
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("gnani_prisma_stt")

# ── Configuration ────────────────────────────────────────────────────────────
GNANI_API_ENDPOINT = "https://api.vachana.ai/stt/v3"

def _load_api_key() -> str:
    key = os.environ.get("GNANI_API_KEY", "")
    if key:
        return key
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("GNANI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""

GNANI_API_KEY = _load_api_key()

# Default parameters optimised for Sanjeevani's Hindi/Hinglish cancer-support
# counselling sessions (noisy telephonic audio, code-mixed speech)
DEFAULT_MODEL         = "prisma-v3.0"
DEFAULT_LANGUAGE_CODE = "indic-auto"   # Auto-detects Hindi, Hinglish, English
DEFAULT_FORMAT        = "codemix"      # Best for Hinglish / code-mixed speech


# ── Core Transcription Functions ─────────────────────────────────────────────

def transcribe_audio_bytes(
    audio_bytes: bytes,
    filename: str = "audio.wav",
    language_code: str = DEFAULT_LANGUAGE_CODE,
    output_format: str = DEFAULT_FORMAT,
    api_key: Optional[str] = None,
) -> dict:
    """
    Transcribe audio from raw bytes using Gnani Prisma v3.0 STT.

    Args:
        audio_bytes:    Raw audio file bytes (WAV, MP3, OGG, FLAC, AAC, M4A)
        filename:       Original filename (used for MIME type hint)
        language_code:  Language or 'indic-auto' for auto-detection
        output_format:  'verbatim' | 'transcribe' | 'codemix' | 'transliterate'
        api_key:        Gnani API key (falls back to GNANI_API_KEY env var)

    Returns:
        dict with keys:
            transcript          (str)   — transcribed text
            detected_language   (str)   — e.g. "hi-IN"
            processing_time     (float) — seconds Gnani took
            success             (bool)
            error               (str | None)
            raw_response        (dict)  — full Gnani JSON response
    """
    key = api_key or GNANI_API_KEY
    if not key:
        return _error_result("GNANI_API_KEY not set. Export it or pass api_key=")

    headers = {"X-API-Key-ID": key}
    files   = {"audio_file": (filename, io.BytesIO(audio_bytes))}
    data    = {
        "model":         DEFAULT_MODEL,
        "language_code": language_code,
        "format":        output_format,
    }

    try:
        logger.info(
            "Calling Gnani Prisma v3.0 STT | lang=%s | format=%s | size=%d bytes",
            language_code, output_format, len(audio_bytes)
        )
        response = requests.post(
            GNANI_API_ENDPOINT,
            headers=headers,
            files=files,
            data=data,
            timeout=30,
        )
        response.raise_for_status()
        body = response.json()

        transcript = body.get("transcript", "")
        logger.info("Gnani STT success | lang=%s | transcript=%r",
                    body.get("detected_language_code"), transcript[:80])

        return {
            "transcript":        transcript,
            "detected_language": body.get("detected_language_code", language_code),
            "processing_time":   body.get("processing_time", 0.0),
            "success":           body.get("success", True),
            "error":             None,
            "raw_response":      body,
        }

    except requests.exceptions.Timeout:
        return _error_result("Gnani API timed out after 30 seconds")
    except requests.exceptions.HTTPError as exc:
        return _error_result(f"Gnani API HTTP error: {exc.response.status_code} {exc.response.text}")
    except requests.exceptions.RequestException as exc:
        return _error_result(f"Gnani API network error: {exc}")
    except Exception as exc:  # noqa: BLE001
        return _error_result(f"Unexpected error: {exc}")


def transcribe_audio_file(
    file_path: str | Path,
    language_code: str = DEFAULT_LANGUAGE_CODE,
    output_format: str = DEFAULT_FORMAT,
    api_key: Optional[str] = None,
) -> dict:
    """
    Transcribe audio from a file path using Gnani Prisma v3.0 STT.
    Convenience wrapper around transcribe_audio_bytes().
    """
    path = Path(file_path)
    if not path.exists():
        return _error_result(f"File not found: {file_path}")
    audio_bytes = path.read_bytes()
    return transcribe_audio_bytes(
        audio_bytes=audio_bytes,
        filename=path.name,
        language_code=language_code,
        output_format=output_format,
        api_key=api_key,
    )


def get_transcript_text(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """
    Minimal helper — returns just the transcript string (empty string on error).
    Drop-in replacement for the existing SpeechRecognition call.

    Usage (replace existing STT call with):
        from gnani_prisma_stt import get_transcript_text
        transcript = get_transcript_text(audio_bytes, original_filename)
    """
    result = transcribe_audio_bytes(audio_bytes, filename)
    if not result["success"]:
        logger.warning("Gnani STT failed: %s", result["error"])
    return result["transcript"]


# ── Internal Helpers ─────────────────────────────────────────────────────────

def _error_result(message: str) -> dict:
    logger.error("Gnani STT error: %s", message)
    return {
        "transcript":        "",
        "detected_language": "",
        "processing_time":   0.0,
        "success":           False,
        "error":             message,
        "raw_response":      {},
    }


# ── Quick Test (run directly: python gnani_prisma_stt.py) ────────────────────

if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python gnani_prisma_stt.py <audio_file> [language_code]")
        print("  Example: python gnani_prisma_stt.py 'python code/benchmark_audio_samples/CASE_1_HINDI_AC....wav' hi-IN")
        print()
        print("Ensure GNANI_API_KEY is set in your environment.")
        sys.exit(1)

    audio_path = sys.argv[1]
    lang       = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_LANGUAGE_CODE

    print(f"\n🔊 Transcribing: {audio_path}")
    print(f"🌐 Language:     {lang}")
    print(f"🤖 Model:        {DEFAULT_MODEL}\n")

    result = transcribe_audio_file(audio_path, language_code=lang)
    print("=" * 60)
    print(f"✅ Success:          {result['success']}")
    print(f"📝 Transcript:       {result['transcript']}")
    print(f"🗣  Detected Lang:   {result['detected_language']}")
    print(f"⏱  Processing Time: {result['processing_time']:.3f}s")
    if result["error"]:
        print(f"❌ Error:           {result['error']}")
    print("=" * 60)
    print("\nFull response JSON:")
    print(json.dumps(result["raw_response"], ensure_ascii=False, indent=2))

