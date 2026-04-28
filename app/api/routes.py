"""
app/api/routes.py
-----------------
STEP 10 — FastAPI Endpoints

Endpoints:
  GET  /health
  POST /predict_window     — single window inference
  POST /predict_batch      — batch inference
  POST /stream_detect      — streaming chunk ingestion
  POST /upload_edf         — upload EDF file for analysis
  POST /stream_simulate    — replay a full EEG recording as a stream
  GET  /alerts             — history of triggered seizure alerts
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import List

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.config import settings
from app.core.logging import logger
from app.schemas.prediction import (
    BatchPredictRequest,
    BatchPredictionResponse,
    EDFUploadResponse,
    HealthResponse,
    DisorderPredictionResponse,
    ParkinsonPredictRequest,
    PredictionResponse,
    StreamChunkRequest,
    StreamResponse,
    WindowPredictRequest,
)
from app.services.inference import InferenceService, simulate_realtime_stream
from app.services.disorder_models import DisorderModelService

router = APIRouter(prefix=settings.API_PREFIX)


# ── Dependency injection ───────────────────────────────────────────────────────
# The InferenceService is created at app startup and injected here.
# Using a module-level reference allows the lifespan handler to set it once.

_service: InferenceService | None = None
_disorder_service: DisorderModelService = DisorderModelService()


def get_service() -> InferenceService:
    if _service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Trigger /health to diagnose.",
        )
    return _service


def set_service(service: InferenceService) -> None:
    global _service
    _service = service


# ── Health ─────────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    tags=["System"],
)
async def health() -> HealthResponse:
    """Returns service health and model load status."""
    from app.models.model_registry import registry

    model_loaded = _service is not None
    device = str(_service.wrapper.device) if model_loaded else "unknown"
    expected_channels = _service.wrapper.model.n_channels if model_loaded else None
    expected_window_size = _service.wrapper.model.window_size if model_loaded else None
    metadata = registry.get_cnn_metadata() if model_loaded else {}
    training_config = metadata.get("training_config", {})
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        device=device,
        version=settings.MODEL_VERSION,
        expected_channels=expected_channels,
        expected_window_size=expected_window_size,
        dataset=training_config.get("dataset"),
    )


# ── Single-window prediction ───────────────────────────────────────────────────

@router.post(
    "/predict_window",
    response_model=PredictionResponse,
    summary="Predict seizure probability for a single EEG window",
    tags=["Inference"],
)
async def predict_window(
    request: WindowPredictRequest,
    service: InferenceService = Depends(get_service),
) -> PredictionResponse:
    """
    Accepts a single EEG window as a nested list and returns a seizure
    probability with an ALERT/CLEAR status.

    **Input shape**: `window[n_channels][window_size]`
    """
    try:
        window = request.as_numpy
        result = service.predict_window(window)
        return PredictionResponse.from_result(result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("predict_window failed: {}", e)
        raise HTTPException(status_code=500, detail="Inference error")


# ── Batch prediction ───────────────────────────────────────────────────────────

@router.post(
    "/predict_batch",
    response_model=BatchPredictionResponse,
    summary="Batch predict over multiple EEG windows",
    tags=["Inference"],
)
async def predict_batch(
    request: BatchPredictRequest,
    service: InferenceService = Depends(get_service),
) -> BatchPredictionResponse:
    """
    Accepts up to 512 windows per request.
    Returns a prediction for each window plus aggregate alert count.

    **Input shape**: `windows[n_windows][n_channels][window_size]`
    """
    try:
        X = request.as_numpy
        results = service.predict_batch(X)
        predictions = [PredictionResponse.from_result(r) for r in results]
        return BatchPredictionResponse(
            predictions=predictions,
            n_alerts=sum(r.is_seizure for r in results),
            n_windows=len(results),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("predict_batch failed: {}", e)
        raise HTTPException(status_code=500, detail="Batch inference error")


# ── Streaming detection ────────────────────────────────────────────────────────

@router.post(
    "/stream_detect",
    response_model=StreamResponse,
    summary="Feed a raw EEG chunk into the streaming buffer",
    tags=["Streaming"],
)
async def stream_detect(
    request: StreamChunkRequest,
    service: InferenceService = Depends(get_service),
) -> StreamResponse:
    """
    Ingests an arbitrary-length EEG chunk into the ring-buffer.
    Returns predictions for all complete windows that became available.

    Call this endpoint repeatedly at your data acquisition rate (e.g. every
    256 samples / 256 Hz = every 1 second).

    **Input shape**: `chunk[n_channels][n_chunk_samples]`
    """
    try:
        chunk   = request.as_numpy
        results = await service.stream_chunk(chunk)
        predictions = [PredictionResponse.from_result(r) for r in results]
        # Remaining buffer length
        buf_remaining = (
            service._buffer.shape[-1]
            if service._buffer is not None
            else 0
        )
        return StreamResponse(
            predictions=predictions,
            buffer_samples_remaining=buf_remaining,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("stream_detect failed: {}", e)
        raise HTTPException(status_code=500, detail="Streaming inference error")


# ── EDF upload ─────────────────────────────────────────────────────────────────

@router.post(
    "/upload_edf",
    response_model=EDFUploadResponse,
    summary="Upload an EDF file and run seizure detection",
    tags=["Data"],
)
async def upload_edf(
    file: UploadFile = File(..., description="EDF file to analyse"),
    service: InferenceService = Depends(get_service),
) -> EDFUploadResponse:
    """
    Accepts an EDF file upload, loads it via MNE, runs windowed inference,
    and returns a summary including seizure annotations if present.
    """
    if not file.filename.lower().endswith(".edf"):
        raise HTTPException(
            status_code=400,
            detail="Only EDF files are accepted.",
        )

    try:
        from app.models.model_registry import registry
        from ml.data_loader.edf_loader import EDFLoader
        from ml.preprocessing.windowing import SlidingWindowSegmenter

        contents = await file.read()

        # Save to a temp file (MNE requires a real file path)
        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)

        metadata = registry.get_cnn_metadata()
        training_config = metadata.get("training_config", {})
        target_sfreq = training_config.get("sfreq")
        window_size = training_config.get(
            "window_size",
            service.wrapper.model.window_size,
        )
        window_stride = training_config.get(
            "window_stride",
            max(window_size // 2, 1),
        )

        loader = EDFLoader(target_sfreq=target_sfreq)
        record = loader.load(tmp_path)
        tmp_path.unlink(missing_ok=True)

        # Run batch inference on all windows
        segmenter = SlidingWindowSegmenter(
            window_size=window_size,
            stride=window_stride,
        )
        windows   = segmenter.segment(record)
        X = np.stack([w.window for w in windows], axis=0)
        _ = service.predict_batch(X)  # results go to alert history

        return EDFUploadResponse(
            filename=file.filename,
            n_channels=len(record.channel_names),
            n_samples=record.n_samples,
            duration_sec=round(record.duration_sec, 2),
            sfreq=record.sfreq,
            n_seizures_annotated=len(record.seizures),
            message=f"Processed {len(windows)} windows. Check /alerts for detections.",
        )

    except ValueError as e:
        actual_channels = locals().get("record").signals.shape[0] if "record" in locals() else None
        expected_channels = service.wrapper.model.n_channels
        expected_window_size = service.wrapper.model.window_size
        detail = str(e)
        if actual_channels is not None and actual_channels != expected_channels:
            detail = (
                f"Uploaded EDF has {actual_channels} channel(s), but the active model expects "
                f"{expected_channels}. The currently loaded checkpoint is compatible with "
                f"{expected_channels}-channel windows of length {expected_window_size}. "
                "Use a matching model or upload data prepared for the active checkpoint."
            )
        raise HTTPException(status_code=422, detail=detail)
    except Exception as e:
        logger.exception("upload_edf failed: {}", e)
        raise HTTPException(status_code=500, detail=f"EDF processing error: {e}")


# ── Alert history ──────────────────────────────────────────────────────────────

@router.get(
    "/alerts",
    summary="Get history of triggered seizure alerts",
    tags=["Monitoring"],
)
async def get_alerts(
    service: InferenceService = Depends(get_service),
) -> dict:
    """Returns the most recent seizure alerts from the in-memory alert log."""
    return {
        "alerts": service.alert_history,
        "total": len(service.alert_history),
    }


# ── Model versions ─────────────────────────────────────────────────────────────

@router.get(
    "/models",
    summary="List available model versions",
    tags=["System"],
)
async def list_models() -> dict:
    from app.models.model_registry import registry
    return registry.list_versions()


@router.post(
    "/predict/alzheimers",
    response_model=DisorderPredictionResponse,
    summary="Predict Alzheimer's class from MRI image",
    tags=["Disorders"],
)
async def predict_alzheimers(file: UploadFile = File(...)) -> DisorderPredictionResponse:
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        raise HTTPException(status_code=400, detail="Upload a valid image file.")
    try:
        with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)
        result = _disorder_service.predict_alzheimers(tmp_path)
        tmp_path.unlink(missing_ok=True)
        return DisorderPredictionResponse(disorder="alzheimers", **result)
    except Exception as e:
        logger.exception("predict_alzheimers failed: {}", e)
        raise HTTPException(status_code=500, detail=f"Alzheimer prediction error: {e}")


@router.post(
    "/predict/parkinsons",
    response_model=DisorderPredictionResponse,
    summary="Predict Parkinson's from tabular voice features",
    tags=["Disorders"],
)
async def predict_parkinsons(request: ParkinsonPredictRequest) -> DisorderPredictionResponse:
    try:
        result = _disorder_service.predict_parkinsons(request.model_dump())
        return DisorderPredictionResponse(disorder="parkinsons", **result)
    except Exception as e:
        logger.exception("predict_parkinsons failed: {}", e)
        raise HTTPException(status_code=500, detail=f"Parkinson prediction error: {e}")


@router.post(
    "/predict/neuro",
    response_model=DisorderPredictionResponse,
    summary="Predict brain tumor/other neuro class from image",
    tags=["Disorders"],
)
async def predict_neuro(file: UploadFile = File(...)) -> DisorderPredictionResponse:
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        raise HTTPException(status_code=400, detail="Upload a valid image file.")
    try:
        with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)
        result = _disorder_service.predict_neuro(tmp_path)
        tmp_path.unlink(missing_ok=True)
        return DisorderPredictionResponse(disorder="neuro", **result)
    except Exception as e:
        logger.exception("predict_neuro failed: {}", e)
        raise HTTPException(status_code=500, detail=f"Neuro prediction error: {e}")


@router.get(
    "/model_info",
    summary="Get metadata for all disorder models",
    tags=["Disorders"],
)
async def model_info() -> dict:
    return _disorder_service.model_info()


@router.get(
    "/dataset_info",
    summary="Get metadata for all datasets",
    tags=["Disorders"],
)
async def dataset_info() -> dict:
    return _disorder_service.dataset_info()


@router.get(
    "/results",
    summary="Get recent disorder prediction results",
    tags=["Disorders"],
)
async def results() -> dict:
    return {"results": _disorder_service.results, "total": len(_disorder_service.results)}
