"""
app/schemas/prediction.py
--------------------------
Pydantic v2 models for all API request/response payloads.

Strict typing ensures that invalid input is caught at the boundary
(FastAPI validation layer) before it reaches any business logic.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Request schemas ────────────────────────────────────────────────────────────

class WindowPredictRequest(BaseModel):
    """
    A single pre-cut EEG window delivered as a nested list.
    Shape: [n_channels][window_size]
    """
    window: List[List[float]] = Field(
        ...,
        description="EEG window: outer list = channels, inner list = samples",
        examples=[[[0.1, 0.2, -0.3] * 85 + [0.0]] * 23],
    )
    sfreq: float = Field(
        default=256.0,
        gt=0,
        description="Sampling frequency in Hz",
    )

    @field_validator("window")
    @classmethod
    def validate_window_shape(cls, v: List[List[float]]) -> List[List[float]]:
        if not v:
            raise ValueError("window must not be empty")
        lengths = {len(ch) for ch in v}
        if len(lengths) > 1:
            raise ValueError("All channels must have the same number of samples")
        return v

    @property
    def as_numpy(self):
        import numpy as np
        return np.array(self.window, dtype=np.float32)


class BatchPredictRequest(BaseModel):
    """
    Multiple windows in one request.
    Shape: [n_windows][n_channels][window_size]
    """
    windows: List[List[List[float]]] = Field(
        ...,
        description="Batch of EEG windows",
        min_length=1,
        max_length=512,
    )
    sfreq: float = Field(default=256.0, gt=0)

    @property
    def as_numpy(self):
        import numpy as np
        return np.array(self.windows, dtype=np.float32)


class StreamChunkRequest(BaseModel):
    """
    A raw EEG chunk for streaming ingestion.
    Shape: [n_channels][n_chunk_samples]  (arbitrary chunk length)
    """
    chunk: List[List[float]] = Field(
        ...,
        description="Raw EEG chunk to append to the streaming buffer",
    )
    sfreq: float = Field(default=256.0, gt=0)

    @property
    def as_numpy(self):
        import numpy as np
        return np.array(self.chunk, dtype=np.float32)


# ── Response schemas ───────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    status: str = Field(
        ...,
        description="'ALERT' if seizure probability >= threshold, else 'CLEAR'",
        examples=["ALERT"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Seizure probability output by the model",
    )
    threshold: float = Field(
        ...,
        description="Decision threshold used for this prediction",
    )
    timestamp: float = Field(
        ...,
        description="Unix epoch seconds at prediction time",
    )
    window_start_sample: Optional[int] = Field(
        default=None,
        description="Absolute sample index of the window start (streaming only)",
    )

    @classmethod
    def from_result(cls, result) -> "PredictionResponse":
        return cls(**result.to_dict())


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    n_alerts: int
    n_windows: int


class StreamResponse(BaseModel):
    predictions: List[PredictionResponse]
    buffer_samples_remaining: int = Field(
        ...,
        description="Unconsumed samples still in the streaming buffer",
    )


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str = "ok"
    model_loaded: bool
    device: str
    version: str
    expected_channels: int | None = None
    expected_window_size: int | None = None
    dataset: str | None = None


class EDFUploadResponse(BaseModel):
    filename: str
    n_channels: int
    n_samples: int
    duration_sec: float
    sfreq: float
    n_seizures_annotated: int
    message: str


class ParkinsonPredictRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class DisorderPredictionResponse(BaseModel):
    disorder: str
    prediction: str
    confidence: float
    classes: List[str] | None = None
    target_value: int | None = None
