"""
tests/test_inference.py
------------------------
Unit tests for InferenceService: streaming buffer, alert dispatch,
single/batch prediction logic.

All tests use a mock ModelWrapper to isolate inference service logic
from the neural network.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.inference import (
    AlertDispatcher,
    InferenceService,
    ModelWrapper,
    PredictionResult,
    simulate_realtime_stream,
)

N_CH  = 4
WIN   = 256
SFREQ = 256.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mock_wrapper(prob: float = 0.1) -> MagicMock:
    """Return a ModelWrapper mock whose predict_window always returns `prob`."""
    wrapper = MagicMock(spec=ModelWrapper)
    wrapper.predict_window.return_value = prob
    wrapper.predict_batch.return_value  = np.full(8, prob, dtype=np.float32)
    wrapper.device = "cpu"
    return wrapper


def _service(prob: float = 0.1, threshold: float = 0.5) -> InferenceService:
    return InferenceService(
        model_wrapper=_mock_wrapper(prob),
        window_size=WIN,
        stride=WIN // 2,
        threshold=threshold,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PredictionResult
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictionResult:

    def test_clear_status(self):
        r = PredictionResult(status="CLEAR", confidence=0.2, threshold=0.5)
        assert not r.is_seizure

    def test_alert_status(self):
        r = PredictionResult(status="ALERT", confidence=0.8, threshold=0.5)
        assert r.is_seizure

    def test_to_dict_keys(self):
        r = PredictionResult(status="CLEAR", confidence=0.1, threshold=0.5)
        d = r.to_dict()
        for key in ("status", "confidence", "threshold", "timestamp"):
            assert key in d, f"Missing key: {key}"

    def test_confidence_rounded_in_dict(self):
        r = PredictionResult(status="CLEAR", confidence=0.123456789, threshold=0.5)
        assert r.to_dict()["confidence"] == round(0.123456789, 4)


# ─────────────────────────────────────────────────────────────────────────────
# AlertDispatcher
# ─────────────────────────────────────────────────────────────────────────────

class TestAlertDispatcher:

    def test_clear_result_not_in_history(self):
        d = AlertDispatcher()
        d.dispatch(PredictionResult(status="CLEAR", confidence=0.1, threshold=0.5))
        assert d.alert_history == []

    def test_alert_result_stored_in_history(self):
        d = AlertDispatcher()
        d.dispatch(PredictionResult(status="ALERT", confidence=0.9, threshold=0.5))
        assert len(d.alert_history) == 1
        assert d.alert_history[0]["status"] == "ALERT"

    def test_history_max_length_respected(self):
        d = AlertDispatcher(max_history=3)
        for _ in range(10):
            d.dispatch(PredictionResult(status="ALERT", confidence=0.9, threshold=0.5))
        # alert_history only includes ALERT results, deque caps at 3 total
        assert len(d.alert_history) <= 3

    def test_mixed_dispatches(self):
        d = AlertDispatcher()
        d.dispatch(PredictionResult(status="CLEAR", confidence=0.1, threshold=0.5))
        d.dispatch(PredictionResult(status="ALERT", confidence=0.9, threshold=0.5))
        d.dispatch(PredictionResult(status="CLEAR", confidence=0.2, threshold=0.5))
        assert len(d.alert_history) == 1


# ─────────────────────────────────────────────────────────────────────────────
# InferenceService.predict_window
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictWindow:

    def test_clear_below_threshold(self):
        svc = _service(prob=0.1, threshold=0.5)
        window = np.random.randn(N_CH, WIN).astype(np.float32)
        result = svc.predict_window(window)
        assert result.status == "CLEAR"
        assert result.confidence == pytest.approx(0.1)

    def test_alert_above_threshold(self):
        svc = _service(prob=0.9, threshold=0.5)
        window = np.random.randn(N_CH, WIN).astype(np.float32)
        result = svc.predict_window(window)
        assert result.status == "ALERT"

    def test_exactly_at_threshold_is_alert(self):
        """Probability == threshold should trigger ALERT (>= comparison)."""
        svc = _service(prob=0.5, threshold=0.5)
        window = np.random.randn(N_CH, WIN).astype(np.float32)
        result = svc.predict_window(window)
        assert result.status == "ALERT"

    def test_wrong_window_size_raises(self):
        svc = _service()
        bad_window = np.random.randn(N_CH, WIN + 10).astype(np.float32)
        with pytest.raises(ValueError, match="window_size"):
            svc.predict_window(bad_window)

    def test_threshold_stored_in_result(self):
        svc = _service(prob=0.1, threshold=0.7)
        window = np.random.randn(N_CH, WIN).astype(np.float32)
        result = svc.predict_window(window)
        assert result.threshold == 0.7


# ─────────────────────────────────────────────────────────────────────────────
# InferenceService.predict_batch
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictBatch:

    def test_returns_correct_count(self):
        n = 8
        wrapper = _mock_wrapper(prob=0.1)
        wrapper.predict_batch.return_value = np.full(n, 0.1, dtype=np.float32)
        svc = InferenceService(
            model_wrapper=wrapper, window_size=WIN, stride=WIN // 2, threshold=0.5
        )
        X = np.random.randn(n, N_CH, WIN).astype(np.float32)
        results = svc.predict_batch(X)
        assert len(results) == n

    def test_all_alert_when_prob_high(self):
        n = 4
        wrapper = _mock_wrapper()
        wrapper.predict_batch.return_value = np.full(n, 0.99, dtype=np.float32)
        svc = InferenceService(
            model_wrapper=wrapper, window_size=WIN, stride=WIN // 2, threshold=0.5
        )
        X = np.random.randn(n, N_CH, WIN).astype(np.float32)
        results = svc.predict_batch(X)
        assert all(r.status == "ALERT" for r in results)

    def test_all_clear_when_prob_low(self):
        n = 4
        wrapper = _mock_wrapper()
        wrapper.predict_batch.return_value = np.full(n, 0.01, dtype=np.float32)
        svc = InferenceService(
            model_wrapper=wrapper, window_size=WIN, stride=WIN // 2, threshold=0.5
        )
        X = np.random.randn(n, N_CH, WIN).astype(np.float32)
        results = svc.predict_batch(X)
        assert all(r.status == "CLEAR" for r in results)


# ─────────────────────────────────────────────────────────────────────────────
# InferenceService.stream_chunk (async)
# ─────────────────────────────────────────────────────────────────────────────

class TestStreamChunk:

    @pytest.mark.asyncio
    async def test_small_chunk_produces_no_predictions(self):
        svc = _service()
        # 64 samples < 256 window → no complete window yet
        chunk = np.random.randn(N_CH, 64).astype(np.float32)
        results = await svc.stream_chunk(chunk)
        assert results == []

    @pytest.mark.asyncio
    async def test_full_window_chunk_produces_prediction(self):
        svc = _service(prob=0.1)
        # Exactly one window worth of data
        chunk = np.random.randn(N_CH, WIN).astype(np.float32)
        results = await svc.stream_chunk(chunk)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_buffer_accumulation(self):
        """Feeding multiple small chunks should eventually produce a prediction."""
        svc = _service(prob=0.1)
        chunk_size = 64
        total_fed = 0
        all_results = []

        for _ in range(5):
            chunk = np.random.randn(N_CH, chunk_size).astype(np.float32)
            results = await svc.stream_chunk(chunk)
            all_results.extend(results)
            total_fed += chunk_size

        # After 5 × 64 = 320 samples, we should have at least one window
        assert len(all_results) >= 1, (
            f"Expected predictions after feeding {total_fed} samples"
        )

    @pytest.mark.asyncio
    async def test_alert_triggered_by_stream(self):
        svc = _service(prob=0.99, threshold=0.5)
        chunk = np.random.randn(N_CH, WIN).astype(np.float32)
        results = await svc.stream_chunk(chunk)
        assert any(r.is_seizure for r in results)

    @pytest.mark.asyncio
    async def test_reset_clears_buffer(self):
        svc = _service()
        # Feed partial chunk
        await svc.stream_chunk(np.random.randn(N_CH, 64).astype(np.float32))
        assert svc._buffer is not None
        svc.reset_stream()
        assert svc._buffer is None
        assert svc._sample_counter == 0

    @pytest.mark.asyncio
    async def test_window_start_sample_increments(self):
        """window_start_sample should track absolute position in the stream."""
        svc = _service(prob=0.1)
        chunk = np.random.randn(N_CH, WIN * 3).astype(np.float32)
        results = await svc.stream_chunk(chunk)

        if len(results) >= 2:
            start0 = results[0].window_start_sample
            start1 = results[1].window_start_sample
            assert start1 > start0, "Later windows must have higher sample index"


# ─────────────────────────────────────────────────────────────────────────────
# Simulation helper
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulateRealtimeStream:

    @pytest.mark.asyncio
    async def test_simulation_processes_all_chunks(self):
        svc = _service(prob=0.1)
        signals = np.random.randn(N_CH, WIN * 10).astype(np.float32)
        results = await simulate_realtime_stream(svc, signals, chunk_size=WIN)
        # 10 windows worth → expect predictions
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_simulation_resets_between_runs(self):
        """Running simulation twice should not accumulate buffer state."""
        svc = _service(prob=0.1)
        signals = np.random.randn(N_CH, WIN * 4).astype(np.float32)

        r1 = await simulate_realtime_stream(svc, signals, chunk_size=WIN)
        r2 = await simulate_realtime_stream(svc, signals, chunk_size=WIN)

        assert len(r1) == len(r2), (
            "Identical signals should produce same number of predictions"
        )
