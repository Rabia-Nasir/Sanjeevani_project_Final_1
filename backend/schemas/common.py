"""
Sanjeevani AI - Common Schemas (Health, Ready, Model Metadata)
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="Service liveness status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Server ISO timestamp")

class ReadyResponse(BaseModel):
    status: str = Field(default="ready", description="Service readiness status")
    models_loaded: Dict[str, bool] = Field(..., description="Readiness flag for each model component")
    device: str = Field(..., description="Active compute device (cpu / cuda)")
    safety_disclaimer: str = Field(..., description="Mandatory non-clinical disclaimer")

class ModelItem(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: str
    task: str
    framework: str
    input_format: str
    output_format: str
    is_active: bool
    version_or_path: str
    calibration_status: str

class ModelListResponse(BaseModel):
    total_models: int
    models: List[ModelItem]
