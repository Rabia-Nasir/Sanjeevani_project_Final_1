"""
Sanjeevani AI - Privacy-Preserving Structured Logger
Ensures that raw patient health texts, voice notes, or sensitive identifiers are never logged.
"""

import logging
import sys
import uuid
from typing import Optional

def setup_logger(name: str = "sanjeevani_api") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [ReqID: %(request_id)s] %(message)s",
                defaults={"request_id": "-"}
            )
        )
        logger.addHandler(handler)
    return logger

logger = setup_logger()

def mask_text_summary(text: Optional[str]) -> str:
    """Returns safe metadata about text without leaking patient health contents."""
    if not text:
        return "empty"
    return f"len={len(text)}_words={len(text.split())}"
