"""
Sanjeevani AI - Backend Configuration
Configuration parameters loaded from environment variables with safe defaults.
"""

import os
from typing import List
from pydantic import BaseModel

class Settings(BaseModel):
    PROJECT_NAME: str = "Sanjeevani AI Affective Computing API"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/v1"
    
    # Server & Port
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Security
    COUNSELLOR_AUTH_KEY: str = os.getenv("COUNSELLOR_AUTH_KEY", "sanjeevani-counsellor-demo-2026")

    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:7860",
        "http://localhost:8000",
        "http://127.0.0.1:7860",
        "http://127.0.0.1:8000",
        "https://sanjeevani-care-ai.loca.lt",
        "*"  # Allows local file-based testing for patient portal.html
    ]
    
    # Request limits
    MAX_TEXT_LENGTH: int = 4000  # Characters
    MAX_AUDIO_BYTES: int = 15 * 1024 * 1024  # 15 MB
    
    # Non-clinical safety notice
    SAFETY_DISCLAIMER: str = (
        "Non-Diagnostic Signal: Affective distress scores and emotional state classifications "
        "are experimental triage prioritization signals designed to assist human counselors. "
        "They do not constitute a psychiatric or clinical oncology diagnosis."
    )

settings = Settings()
