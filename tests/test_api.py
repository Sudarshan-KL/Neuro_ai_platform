"""
tests/test_api.py
-----------------
Integration tests for all FastAPI endpoints.

Uses httpx.AsyncClient with a dependency-override pattern to inject a
mock InferenceService — no trained model required for CI.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings
from app.main import create_app
from app.services.inference import InferenceService, PredictionResult


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

N_CHANNELS   = 4
WINDOW_SIZE  = 256
SFREQ        = 256.0


def _make_window_payload(n_channels: int = N_CHANNELS, size: int = WINDOW_SIZE) -> dict:
    """JSON-serialisable window payload."""
    return {
        "window": np.random.randn(n_channels, size).tolist(),
        "sfreq": SFREQ,
    }


def _make_batch_payload(n: int = 3) -> dict:
    return {
        "windows": np.random.randn(n, N_CHANNELS, WINDOW_SIZE).tolist(),
        "sfreq": SFREQ,
    }


def _make_chunk_payload(chunk_samples: int = 128) -> dict:
    return {
        "chunk": np.random.randn(N_CHANNELS, chunk_samples).tolist(),
        "sfreq": SFREQ,
    }


def _make_clear_result() -> PredictionResult:
    return PredictionResult(status="CLEAR", confidence=0.1, threshold=0.5)


def _make_alert_result() -> PredictionResult:
    return PredictionResult(status="ALERT", confidence=0.91, threshold=0.5)


def _mock_service(
    window_result: PredictionResult | None = None,
    batch_results: List[PredictionResult] | None = None,
    stream_results: List[PredictionResult] | None = None,
) -> MagicMock:
    """Build a fully-mocked InferenceService."""
    svc = MagicMock(spec=InferenceService)
    svc.predict_window.return_value  = window_result or _make_clear_result()
    svc.predict_batch.return_value   = batch_results or [_make_clear_result()]
    svc.stream_chunk                 = AsyncMock(return_value=stream_results or [])
    svc.alert_history                = []
    svc._buffer                      = None
    svc.wrapper.device               = "cpu"
    return svc


@pytest.fixture
def app():
    return create_app()


@pytest_asyncio.fixture
async def client(app):
    """AsyncClient wired to the FastAPI test app."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture(autouse=True)
def inject_mock_service(app):
    """
    Override the service dependency for every test so no real model is needed.
    """
    from app.api import routes as r
    mock_svc = _mock_service()
    r.set_service(mock_svc)
    yield mock_svc
    r.set_service(None)


# ─────────────────────────────────────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────────────────────────────────────

class TestHealth:

    @pytest.mark.asyncio
    async def test_health_ok(self, client):
        resp = await client.get(f"{settings.API_PREFIX}/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True

    @pytest.mark.asyncio
    async def test_health_no_model(self, app, client):
        from app.api import routes as r
        r.set_service(None)
        resp = await client.get(f"{settings.API_PREFIX}/health")
        body = resp.json()
        assert body["model_loaded"] is False
        # Restore
        r.set_service(_mock_service())


# ─────────────────────────────────────────────────────────────────────────────
# /predict_window
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictWindow:

    @pytest.mark.asyncio
    async def test_clear_prediction(self, client, inject_mock_service):
        inject_mock_service.predict_window.return_value = _make_clear_result()
        resp = await client.post(
            f"{settings.API_PREFIX}/predict_window",
            json=_make_window_payload(),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "CLEAR"
        assert 0.0 <= body["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_alert_prediction(self, client, inject_mock_service):
        inject_mock_service.predict_window.return_value = _make_alert_result()
        resp = await client.post(
            f"{settings.API_PREFIX}/predict_window",
            json=_make_window_payload(),
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ALERT"
        assert resp.json()["confidence"] > 0.5

    @pytest.mark.asyncio
    async def test_missing_window_field(self, client):
        resp = await client.post(
            f"{settings.API_PREFIX}/predict_window",
            json={"sfreq": 256.0},   # no 'window'
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_empty_window_rejected(self, client):
        resp = await client.post(
            f"{settings.API_PREFIX}/predict_window",
            json={"window": [], "sfreq": 256.0},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_mismatched_channel_lengths_rejected(self, client):
        bad_window = [[0.1] * 256, [0.2] * 128]   # channels have different lengths
        resp = await client.post(
            f"{settings.API_PREFIX}/predict_window",
            json={"window": bad_window, "sfreq": 256.0},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_sfreq_rejected(self, client):
        resp = await client.post(
            f"{settings.API_PREFIX}/predict_window",
            json={**_make_window_payload(), "sfreq": -1},
        )
        assert resp.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# /predict_batch
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictBatch:

    @pytest.mark.asyncio
    async def test_batch_returns_all_predictions(self, client, inject_mock_service):
        n = 5
        inject_mock_service.predict_batch.return_value = [_make_clear_result()] * n
        resp = await client.post(
            f"{settings.API_PREFIX}/predict_batch",
            json=_make_batch_payload(n=n),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_windows"] == n
        assert len(body["predictions"]) == n

    @pytest.mark.asyncio
    async def test_batch_alert_count(self, client, inject_mock_service):
        results = [_make_alert_result(), _make_clear_result(), _make_alert_result()]
        inject_mock_service.predict_batch.return_value = results
        resp = await client.post(
            f"{settings.API_PREFIX}/predict_batch",
            json=_make_batch_payload(n=3),
        )
        assert resp.json()["n_alerts"] == 2

    @pytest.mark.asyncio
    async def test_empty_batch_rejected(self, client):
        resp = await client.post(
            f"{settings.API_PREFIX}/predict_batch",
            json={"windows": [], "sfreq": 256.0},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_oversized_batch_rejected(self, client):
        # 513 windows > max_length=512
        resp = await client.post(
            f"{settings.API_PREFIX}/predict_batch",
            json={
                "windows": [[[0.0] * WINDOW_SIZE] * N_CHANNELS] * 513,
                "sfreq": SFREQ,
            },
        )
        assert resp.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# /stream_detect
# ─────────────────────────────────────────────────────────────────────────────

class TestStreamDetect:

    @pytest.mark.asyncio
    async def test_stream_no_complete_window_yet(self, client, inject_mock_service):
        """Small chunk — no complete window ready yet."""
        inject_mock_service.stream_chunk = AsyncMock(return_value=[])
        inject_mock_service._buffer = None
        resp = await client.post(
            f"{settings.API_PREFIX}/stream_detect",
            json=_make_chunk_payload(chunk_samples=64),
        )
        assert resp.status_code == 200
        assert resp.json()["predictions"] == []

    @pytest.mark.asyncio
    async def test_stream_returns_predictions(self, client, inject_mock_service):
        inject_mock_service.stream_chunk = AsyncMock(
            return_value=[_make_clear_result(), _make_alert_result()]
        )
        inject_mock_service._buffer = np.zeros((N_CHANNELS, 50))
        resp = await client.post(
            f"{settings.API_PREFIX}/stream_detect",
            json=_make_chunk_payload(chunk_samples=512),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["predictions"]) == 2

    @pytest.mark.asyncio
    async def test_stream_alert_in_predictions(self, client, inject_mock_service):
        inject_mock_service.stream_chunk = AsyncMock(
            return_value=[_make_alert_result()]
        )
        inject_mock_service._buffer = None
        resp = await client.post(
            f"{settings.API_PREFIX}/stream_detect",
            json=_make_chunk_payload(chunk_samples=256),
        )
        assert resp.status_code == 200
        preds = resp.json()["predictions"]
        assert any(p["status"] == "ALERT" for p in preds)


# ─────────────────────────────────────────────────────────────────────────────
# /alerts
# ─────────────────────────────────────────────────────────────────────────────

class TestAlerts:

    @pytest.mark.asyncio
    async def test_empty_alert_history(self, client, inject_mock_service):
        inject_mock_service.alert_history = []
        resp = await client.get(f"{settings.API_PREFIX}/alerts")
        assert resp.status_code == 200
        body = resp.json()
        assert body["alerts"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_alert_history_populated(self, client, inject_mock_service):
        inject_mock_service.alert_history = [
            _make_alert_result().to_dict(),
            _make_alert_result().to_dict(),
        ]
        resp = await client.get(f"{settings.API_PREFIX}/alerts")
        assert resp.json()["total"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# /upload_edf  (mocked at the loader level)
# ─────────────────────────────────────────────────────────────────────────────

class TestUploadEDF:

    @pytest.mark.asyncio
    async def test_non_edf_file_rejected(self, client):
        resp = await client.post(
            f"{settings.API_PREFIX}/upload_edf",
            files={"file": ("report.pdf", b"fake content", "application/pdf")},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_edf_upload_processes_file(self, client, inject_mock_service):
        """
        Mock the EDFLoader so the test doesn't need a real EDF file.
        """
        from ml.data_loader.edf_loader import EEGRecord

        fake_record = EEGRecord(
            file_path=Path("test.edf"),
            signals=np.random.randn(N_CHANNELS, WINDOW_SIZE * 4).astype(np.float32),
            channel_names=[f"CH{i}" for i in range(N_CHANNELS)],
            sfreq=SFREQ,
            n_samples=WINDOW_SIZE * 4,
            duration_sec=4.0,
            seizures=[],
        )
        inject_mock_service.predict_batch.return_value = [_make_clear_result()]

        with patch("app.api.routes.EDFLoader") as mock_loader_cls:
            mock_loader_cls.return_value.load.return_value = fake_record
            resp = await client.post(
                f"{settings.API_PREFIX}/upload_edf",
                files={"file": ("test.edf", b"fake edf bytes", "application/octet-stream")},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["filename"] == "test.edf"
        assert body["n_channels"] == N_CHANNELS
        assert body["sfreq"] == SFREQ
