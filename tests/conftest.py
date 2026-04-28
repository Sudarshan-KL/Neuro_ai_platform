"""
tests/conftest.py
-----------------
Shared pytest fixtures and configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repo root is on sys.path for all tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Shared constants ───────────────────────────────────────────────────────────

N_CHANNELS  = 4
WINDOW_SIZE = 256
SFREQ       = 256.0
BATCH_SIZE  = 8


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def rng():
    """Seeded numpy RNG for reproducible test data."""
    return np.random.default_rng(42)


@pytest.fixture
def random_window(rng):
    return rng.standard_normal((N_CHANNELS, WINDOW_SIZE)).astype(np.float32)


@pytest.fixture
def random_batch(rng):
    return rng.standard_normal((BATCH_SIZE, N_CHANNELS, WINDOW_SIZE)).astype(np.float32)
