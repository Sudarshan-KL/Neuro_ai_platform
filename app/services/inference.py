"""
app/services/inference.py
--------------------------
STEP 8 — Real-Time Streaming Simulation
STEP 9 — Alert System

The InferenceService maintains a sliding ring-buffer fed by incoming raw EEG
chunks.  When the buffer fills one window, it runs the CNN and emits an alert
if the seizure probability exceeds the configured threshold.

Thread-safety: the buffer uses a deque protected by asyncio.Lock so the
FastAPI event loop never blocks.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import numpy as np
import torch

from app.core.config import settings
from app.core.logging import logger
from ml.features.feature_extractor import FeatureExtractor, FeatureMode
from ml.training.cnn_model import SeizureCNN


# ── Alert / prediction result ──────────────────────────────────────────────────

@dataclass
class PredictionResult:
    status: str           # "ALERT" | "CLEAR"
    confidence: float     # seizure probability in [0, 1]
    threshold: float
    timestamp: float = field(default_factory=time.time)
    window_start_sample: Optional[int] = None

    @property
    def is_seizure(self) -> bool:
        return self.status == "ALERT"

    def to_dict(self) -> dict:
        return {
            "status":               self.status,
            "confidence":           round(self.confidence, 4),
            "threshold":            self.threshold,
            "timestamp":            self.timestamp,
            "window_start_sample":  self.window_start_sample,
        }


# ── Alert dispatcher ───────────────────────────────────────────────────────────

class AlertDispatcher:
    """
    Extensible alert system.  In production this would push to:
    - WebSocket clients (bedside monitor)
    - Hospital paging system API
    - EHR event bus

    For now it logs with a clear ALERT marker and stores the last N alerts.
    """

    def __init__(self, max_history: int = 100) -> None:
        self._history: Deque[PredictionResult] = deque(maxlen=max_history)

    def dispatch(self, result: PredictionResult) -> None:
        self._history.append(result)
        if result.is_seizure:
            logger.warning(
                "🚨 SEIZURE ALERT | confidence={:.4f} | threshold={:.2f} | ts={:.2f}",
                result.confidence,
                result.threshold,
                result.timestamp,
            )
        else:
            logger.debug("Clear | confidence={:.4f}", result.confidence)

    @property
    def alert_history(self) -> List[dict]:
        return [r.to_dict() for r in self._history if r.is_seizure]


# ── Model wrapper ──────────────────────────────────────────────────────────────

class ModelWrapper:
    """
    Thin wrapper that normalises inputs and runs the CNN.
    Keeps the inference path decoupled from model architecture details.
    """

    def __init__(
        self,
        model: SeizureCNN,
        device: Optional[str] = None,
    ) -> None:
        self.device    = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model     = model.to(self.device)
        self.model.eval()
        self.extractor = FeatureExtractor(mode=FeatureMode.DL)

    def predict_window(self, window: np.ndarray) -> float:
        """
        Parameters
        ----------
        window : np.ndarray, shape (n_channels, window_size)

        Returns
        -------
        Seizure probability in [0, 1].
        """
        if window.ndim != 2:
            raise ValueError(f"Expected 2D window (channels, samples), got shape {window.shape}")
        if window.shape[0] != self.model.n_channels:
            raise ValueError(
                f"Expected {self.model.n_channels} channel(s), got {window.shape[0]}"
            )
        if window.shape[1] != self.model.window_size:
            raise ValueError(
                f"Expected window_size={self.model.window_size}, got {window.shape[1]}"
            )
        # z-score normalise: shape (1, C, T)
        normed = self.extractor.transform(window[np.newaxis])
        tensor = torch.from_numpy(normed).float().to(self.device)
        with torch.no_grad():
            prob = self.model(tensor).item()
        return float(prob)

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : np.ndarray, shape (N, C, T)

        Returns
        -------
        probs : np.ndarray, shape (N,)
        """
        if X.ndim != 3:
            raise ValueError(f"Expected 3D batch (N, channels, samples), got shape {X.shape}")
        if X.shape[1] != self.model.n_channels:
            raise ValueError(
                f"Expected {self.model.n_channels} channel(s), got {X.shape[1]}"
            )
        if X.shape[2] != self.model.window_size:
            raise ValueError(
                f"Expected window_size={self.model.window_size}, got {X.shape[2]}"
            )
        normed = self.extractor.transform(X)
        tensor = torch.from_numpy(normed).float().to(self.device)
        with torch.no_grad():
            probs = self.model(tensor).cpu().numpy()
        return probs


# ── Inference service ──────────────────────────────────────────────────────────

class InferenceService:
    """
    Production inference service with:
    * Single-window prediction
    * Batch prediction
    * Real-time streaming simulation with ring-buffer

    A single instance of this class is shared across all FastAPI requests
    (application-level singleton mounted at startup).
    """

    def __init__(
        self,
        model_wrapper: ModelWrapper,
        window_size: int = settings.WINDOW_SIZE,
        stride: int = settings.WINDOW_STRIDE,
        threshold: float = settings.SEIZURE_THRESHOLD,
    ) -> None:
        self.wrapper     = model_wrapper
        self.window_size = window_size
        self.stride      = stride
        self.threshold   = threshold
        self.dispatcher  = AlertDispatcher()

        # Streaming state
        self._buffer: Optional[np.ndarray] = None   # (n_channels, buffer_samples)
        self._sample_counter: int          = 0
        self._lock                         = asyncio.Lock()

    # ── Single-window prediction ──────────────────────────────────────────────

    def predict_window(self, window: np.ndarray) -> PredictionResult:
        """
        Predict seizure probability for a single pre-cut window.

        Parameters
        ----------
        window : (n_channels, window_size) float32 array

        Returns
        -------
        PredictionResult with status and confidence.
        """
        if window.shape[-1] != self.window_size:
            raise ValueError(
                f"Expected window_size={self.window_size}, "
                f"got {window.shape[-1]}"
            )

        prob = self.wrapper.predict_window(window)
        result = PredictionResult(
            status="ALERT" if prob >= self.threshold else "CLEAR",
            confidence=prob,
            threshold=self.threshold,
        )
        self.dispatcher.dispatch(result)
        logger.info(
            "predict_window | shape={} | prob={:.4f} | status={}",
            window.shape, prob, result.status,
        )
        return result

    # ── Batch prediction ──────────────────────────────────────────────────────

    def predict_batch(self, X: np.ndarray) -> List[PredictionResult]:
        """
        Predict over a batch of windows.

        Parameters
        ----------
        X : (N, n_channels, window_size)

        Returns
        -------
        List[PredictionResult]
        """
        probs = self.wrapper.predict_batch(X)
        results = []
        for i, prob in enumerate(probs):
            r = PredictionResult(
                status="ALERT" if prob >= self.threshold else "CLEAR",
                confidence=float(prob),
                threshold=self.threshold,
            )
            results.append(r)
            self.dispatcher.dispatch(r)

        n_alerts = sum(r.is_seizure for r in results)
        logger.info(
            "predict_batch | n={} | alerts={} ({:.1f}%)",
            len(results), n_alerts, 100.0 * n_alerts / max(len(results), 1),
        )
        return results

    # ── Real-time streaming simulation (STEP 8) ───────────────────────────────

    async def stream_chunk(self, chunk: np.ndarray) -> List[PredictionResult]:
        """
        Feed a raw EEG chunk into the ring-buffer and emit predictions for
        each complete window that becomes available.

        This simulates a real bedside monitor feeding 1-second chunks at
        256 samples/s.  Each call may return 0, 1, or many PredictionResults
        depending on how many full windows are now available.

        Parameters
        ----------
        chunk : (n_channels, n_samples_in_chunk) — arbitrary length chunk

        Returns
        -------
        List of PredictionResult (may be empty).
        """
        async with self._lock:
            if chunk.ndim != 2:
                raise ValueError(f"Expected 2D chunk (channels, samples), got shape {chunk.shape}")
            if chunk.shape[0] != self.wrapper.model.n_channels:
                raise ValueError(
                    f"Expected {self.wrapper.model.n_channels} channel(s), got {chunk.shape[0]}"
                )
            # Initialise buffer on first call
            if self._buffer is None:
                n_channels = chunk.shape[0]
                self._buffer = np.zeros(
                    (n_channels, self.window_size),
                    dtype=np.float32,
                )

            # Append chunk to buffer
            self._buffer = np.concatenate([self._buffer, chunk], axis=-1)

            results: List[PredictionResult] = []
            start = 0

            # Slide over buffer to extract complete windows
            while start + self.window_size <= self._buffer.shape[-1]:
                window = self._buffer[:, start : start + self.window_size]
                prob   = self.wrapper.predict_window(window)
                r = PredictionResult(
                    status="ALERT" if prob >= self.threshold else "CLEAR",
                    confidence=prob,
                    threshold=self.threshold,
                    window_start_sample=self._sample_counter + start,
                )
                results.append(r)
                self.dispatcher.dispatch(r)
                start += self.stride

            # Retain only unconsumed samples
            self._buffer = self._buffer[:, start:]
            self._sample_counter += start

            return results

    def reset_stream(self) -> None:
        """Reset streaming state (call between patients / sessions)."""
        self._buffer         = None
        self._sample_counter = 0
        logger.info("Streaming buffer reset.")

    @property
    def alert_history(self) -> List[dict]:
        return self.dispatcher.alert_history


# ── Simulation helper ──────────────────────────────────────────────────────────

async def simulate_realtime_stream(
    service: InferenceService,
    signals: np.ndarray,
    chunk_size: int = 256,   # 1-second chunks at 256 Hz
    sleep_seconds: float = 0.0,  # set > 0 to slow down for demos
) -> List[PredictionResult]:
    """
    Simulate a real-time EEG stream by feeding `signals` in `chunk_size`
    chunks.  Useful for integration tests and live demos.

    Parameters
    ----------
    service       : InferenceService instance
    signals       : (n_channels, n_samples)
    chunk_size    : Samples per "heartbeat"
    sleep_seconds : Artificial delay between chunks

    Returns
    -------
    All PredictionResults emitted during the simulation.
    """
    service.reset_stream()
    all_results: List[PredictionResult] = []
    n_chunks = (signals.shape[-1] + chunk_size - 1) // chunk_size

    logger.info(
        "Starting real-time simulation | n_chunks={} | chunk_size={}",
        n_chunks, chunk_size,
    )

    for i in range(n_chunks):
        chunk = signals[:, i * chunk_size : (i + 1) * chunk_size]
        results = await service.stream_chunk(chunk)
        all_results.extend(results)
        if sleep_seconds > 0:
            await asyncio.sleep(sleep_seconds)

    logger.info(
        "Simulation complete | total_windows={} | alerts={}",
        len(all_results), sum(r.is_seizure for r in all_results),
    )
    return all_results
