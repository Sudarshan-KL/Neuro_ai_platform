"""
ml/features/feature_extractor.py
---------------------------------
STEP 5 — Feature Engineering

Mode A (ML): Hand-crafted statistical + spectral features per window.
Mode B (DL): Raw normalised signal tensor — no feature extraction.

All feature functions are pure (no side effects) and operate on a single
window of shape (n_channels, window_size).
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import kurtosis, skew

from app.core.config import settings
from app.core.logging import logger


class FeatureMode(str, Enum):
    ML = "ml"   # hand-crafted features → flat vector
    DL = "dl"   # raw window tensor


# ── Per-channel statistical features ──────────────────────────────────────────

def _statistical_features(channel: np.ndarray) -> np.ndarray:
    """
    Compute 6 time-domain statistical features for a single channel.

    Returns: [mean, std, variance, skewness, kurtosis, peak-to-peak]
    """
    return np.array([
        channel.mean(),
        channel.std(),
        channel.var(),
        float(skew(channel)),
        float(kurtosis(channel)),
        channel.max() - channel.min(),
    ], dtype=np.float32)


def _energy_features(channel: np.ndarray) -> np.ndarray:
    """
    Signal energy and Hjorth parameters (mobility, complexity).

    Hjorth parameters are classic EEG descriptors widely used in clinical BCI.
    """
    energy   = float(np.sum(channel ** 2))
    diff1    = np.diff(channel)
    diff2    = np.diff(diff1)
    var0     = np.var(channel)  + 1e-12
    var1     = np.var(diff1)    + 1e-12
    var2     = np.var(diff2)    + 1e-12
    mobility   = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / mobility if mobility > 0 else 0.0
    return np.array([energy, mobility, complexity], dtype=np.float32)


def _spectral_features(
    channel: np.ndarray,
    sfreq: float,
) -> np.ndarray:
    """
    Frequency-domain features:
    * Absolute band power: delta, theta, alpha, beta, gamma
    * Spectral entropy (Shannon entropy of the normalised PSD)
    * Spectral edge frequency (95 % of power)

    Clinical relevance: seizures produce characteristic power shifts in
    theta/beta bands and distinctive high-frequency activity.
    """
    # Welch's method gives a stable PSD estimate
    freqs, psd = sp_signal.welch(channel, fs=sfreq, nperseg=min(len(channel), 128))

    def band_power(lo: float, hi: float) -> float:
        idx = np.logical_and(freqs >= lo, freqs < hi)
        return float(np.trapz(psd[idx], freqs[idx])) if idx.any() else 0.0

    bands = [
        band_power(0.5,  4.0),   # delta
        band_power(4.0,  8.0),   # theta
        band_power(8.0, 13.0),   # alpha
        band_power(13.0, 30.0),  # beta
        band_power(30.0, 60.0),  # gamma
    ]

    # Spectral entropy
    psd_norm = psd / (psd.sum() + 1e-12)
    spec_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))

    # Spectral edge frequency (SEF95)
    cumulative = np.cumsum(psd) / (psd.sum() + 1e-12)
    sef95_idx  = np.searchsorted(cumulative, 0.95)
    sef95      = float(freqs[min(sef95_idx, len(freqs) - 1)])

    return np.array(bands + [spec_entropy, sef95], dtype=np.float32)


def extract_window_features(
    window: np.ndarray,
    sfreq: float = settings.SAMPLING_FREQ,
) -> np.ndarray:
    """
    Extract a flat feature vector from a single EEG window.

    Parameters
    ----------
    window : np.ndarray, shape (n_channels, window_size)
    sfreq  : Sampling frequency in Hz

    Returns
    -------
    feature_vector : np.ndarray, shape (n_features,)
    Feature count = n_channels × (6 stat + 3 energy + 7 spectral) = n_channels × 16
    """
    channel_features: List[np.ndarray] = []
    for ch in window:
        ch_feat = np.concatenate([
            _statistical_features(ch),
            _energy_features(ch),
            _spectral_features(ch, sfreq),
        ])
        channel_features.append(ch_feat)

    # Flatten all channels into a 1-D vector
    return np.concatenate(channel_features)


# ── Batch feature extraction ───────────────────────────────────────────────────

class FeatureExtractor:
    """
    Transforms a batch of EEG windows into either:
    - Mode A: flat ML feature matrix, shape (N, n_features)
    - Mode B: normalised 3-D tensor, shape (N, C, T)
    """

    def __init__(
        self,
        mode: FeatureMode = FeatureMode.DL,
        sfreq: float = settings.SAMPLING_FREQ,
    ) -> None:
        self.mode  = mode
        self.sfreq = sfreq

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : np.ndarray, shape (N, C, T)

        Returns
        -------
        Transformed array depending on mode.
        """
        if self.mode == FeatureMode.ML:
            return self._extract_ml_features(X)
        else:
            return self._normalise_for_dl(X)

    def _extract_ml_features(self, X: np.ndarray) -> np.ndarray:
        """Vectorised feature extraction over all windows."""
        logger.info("Extracting ML features from {} windows …", len(X))
        features = np.stack(
            [extract_window_features(w, self.sfreq) for w in X],
            axis=0,
        )
        logger.info("Feature matrix shape: {}", features.shape)
        return features

    @staticmethod
    def _normalise_for_dl(X: np.ndarray) -> np.ndarray:
        """
        Per-window, per-channel z-score normalisation.
        Prevents extreme µV values from dominating gradient updates.
        """
        mu    = X.mean(axis=-1, keepdims=True)
        sigma = X.std(axis=-1, keepdims=True) + 1e-8
        return ((X - mu) / sigma).astype(np.float32)

    @property
    def feature_names(self) -> Optional[List[str]]:
        """Return feature names for SHAP explainability (ML mode only)."""
        if self.mode != FeatureMode.ML:
            return None

        stat_names    = ["mean", "std", "var", "skew", "kurt", "ptp"]
        energy_names  = ["energy", "hjorth_mob", "hjorth_comp"]
        spectral_names = ["delta", "theta", "alpha", "beta", "gamma",
                          "spec_entropy", "sef95"]
        per_ch = stat_names + energy_names + spectral_names

        # We don't know n_channels at init, so return the per-channel template
        return per_ch
