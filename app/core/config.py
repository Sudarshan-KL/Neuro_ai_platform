"""
app/core/config.py
------------------
Centralised configuration using environment variables with sensible defaults.
All tuneable constants live here — never scatter magic numbers across the codebase.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Project paths ────────────────────────────────────────────────────────
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODEL_DIR: Path = PROJECT_ROOT / "saved_models"
    LOG_DIR: Path = PROJECT_ROOT / "logs"
    FRONTEND_DIR: Path = PROJECT_ROOT / "frontend"

    # ── Bonn (Epilepsy-EEG) dataset ───────────────────────────────────────────
    # Path to the folder containing A/ B/ C/ D/ E/ sub-directories
    BONN_DATA_DIR: Path = PROJECT_ROOT / "Epilepsy-EEG"

    # ── EEG / signal — CHB-MIT defaults (kept for backward compat) ──────────
    SAMPLING_FREQ: int = 256          # Hz — CHB-MIT default
    WINDOW_SIZE: int = 256            # samples per window (1 s @ 256 Hz)
    WINDOW_STRIDE: int = 128          # 50 % overlap
    N_CHANNELS: int = 23              # CHB-MIT channels

    # ── EEG / signal — Bonn dataset overrides ────────────────────────────────
    BONN_SFREQ: float = 173.6         # Hz
    BONN_WINDOW_SIZE: int = 173       # ~1 s window at 173.6 Hz
    BONN_WINDOW_STRIDE: int = 87      # ~50 % overlap
    BONN_N_CHANNELS: int = 1          # single-channel recordings

    # ── Training ─────────────────────────────────────────────────────────────
    TEST_SPLIT: float = 0.2
    RANDOM_SEED: int = 42
    BATCH_SIZE: int = 64
    MAX_EPOCHS: int = 30
    LEARNING_RATE: float = 1e-3

    # ── Inference ────────────────────────────────────────────────────────────
    SEIZURE_THRESHOLD: float = 0.5    # probability above which ALERT fires
    MODEL_NAME: str = "cnn_seizure_detector"
    MODEL_VERSION: str = "v1"

    # ── API ───────────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    CORS_ORIGINS: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ]
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def model_path(self) -> Path:
        return self.MODEL_DIR / f"{self.MODEL_NAME}_{self.MODEL_VERSION}.pt"

    @property
    def rf_model_path(self) -> Path:
        return self.MODEL_DIR / f"rf_seizure_detector_{self.MODEL_VERSION}.pkl"

    def ensure_dirs(self) -> None:
        for d in (self.DATA_DIR, self.MODEL_DIR, self.LOG_DIR, self.FRONTEND_DIR):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
