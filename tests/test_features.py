"""
tests/test_features.py
-----------------------
Unit tests for feature engineering (both ML and DL modes).

All tests use synthetic numpy arrays — no EDF files or trained models needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.features.feature_extractor import (
    FeatureExtractor,
    FeatureMode,
    _energy_features,
    _spectral_features,
    _statistical_features,
    extract_window_features,
)


N_CH   = 4
WIN    = 256
SFREQ  = 256.0
RNG    = np.random.default_rng(42)


def _random_window(n_ch: int = N_CH, win: int = WIN) -> np.ndarray:
    return RNG.standard_normal((n_ch, win)).astype(np.float32)


def _random_batch(n: int = 8, n_ch: int = N_CH, win: int = WIN) -> np.ndarray:
    return RNG.standard_normal((n, n_ch, win)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Per-channel feature functions
# ─────────────────────────────────────────────────────────────────────────────

class TestStatisticalFeatures:

    def test_output_shape(self):
        ch = RNG.standard_normal(WIN).astype(np.float32)
        feat = _statistical_features(ch)
        assert feat.shape == (6,), f"Expected (6,), got {feat.shape}"

    def test_dtype_float32(self):
        ch = RNG.standard_normal(WIN).astype(np.float32)
        assert _statistical_features(ch).dtype == np.float32

    def test_mean_correct(self):
        ch = np.ones(WIN, dtype=np.float32) * 5.0
        feat = _statistical_features(ch)
        assert abs(feat[0] - 5.0) < 1e-4, "mean should be 5.0"

    def test_std_zero_for_constant(self):
        ch = np.ones(WIN, dtype=np.float32)
        feat = _statistical_features(ch)
        assert abs(feat[1]) < 1e-4, "std of constant array should be ~0"

    def test_ptp_correct(self):
        ch = np.linspace(-10, 10, WIN, dtype=np.float32)
        feat = _statistical_features(ch)
        assert abs(feat[5] - 20.0) < 0.1, "peak-to-peak should be ~20"


class TestEnergyFeatures:

    def test_output_shape(self):
        ch = RNG.standard_normal(WIN).astype(np.float32)
        feat = _energy_features(ch)
        assert feat.shape == (3,)

    def test_energy_positive(self):
        ch = RNG.standard_normal(WIN).astype(np.float32)
        assert _energy_features(ch)[0] > 0

    def test_zero_signal_energy_is_zero(self):
        ch = np.zeros(WIN, dtype=np.float32)
        feat = _energy_features(ch)
        assert feat[0] == pytest.approx(0.0, abs=1e-6)

    def test_hjorth_values_nonnegative(self):
        ch = RNG.standard_normal(WIN).astype(np.float32)
        feat = _energy_features(ch)
        assert feat[1] >= 0, "Hjorth mobility must be non-negative"
        assert feat[2] >= 0, "Hjorth complexity must be non-negative"


class TestSpectralFeatures:

    def test_output_shape(self):
        ch = RNG.standard_normal(WIN).astype(np.float32)
        feat = _spectral_features(ch, SFREQ)
        # 5 bands + spectral entropy + SEF95 = 7
        assert feat.shape == (7,)

    def test_band_powers_nonnegative(self):
        ch = RNG.standard_normal(WIN).astype(np.float32)
        feat = _spectral_features(ch, SFREQ)
        assert all(v >= 0 for v in feat[:5]), "band powers must be non-negative"

    def test_spectral_entropy_positive(self):
        ch = RNG.standard_normal(WIN).astype(np.float32)
        feat = _spectral_features(ch, SFREQ)
        assert feat[5] > 0

    def test_sef95_within_nyquist(self):
        ch = RNG.standard_normal(WIN).astype(np.float32)
        feat = _spectral_features(ch, SFREQ)
        nyquist = SFREQ / 2
        assert 0 < feat[6] <= nyquist, f"SEF95={feat[6]} outside (0, {nyquist}]"

    def test_dominant_frequency_reflected_in_bands(self):
        """
        A pure 10 Hz sine should have most energy in the alpha band (8–13 Hz).
        """
        t  = np.linspace(0, WIN / SFREQ, WIN, endpoint=False)
        ch = np.sin(2 * np.pi * 10 * t).astype(np.float32)
        feat = _spectral_features(ch, SFREQ)
        # feat[2] = alpha power, should dominate over delta[0]/theta[1]/beta[3]/gamma[4]
        alpha_power = feat[2]
        other_powers = [feat[0], feat[1], feat[3], feat[4]]
        assert alpha_power > max(other_powers), (
            f"10 Hz sine: alpha={alpha_power:.4f} should dominate, got {other_powers}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Full window feature extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractWindowFeatures:

    def test_output_shape(self):
        window = _random_window()
        feat = extract_window_features(window, SFREQ)
        # N_CH channels × 16 features each
        expected_len = N_CH * (6 + 3 + 7)
        assert feat.shape == (expected_len,), (
            f"Expected ({expected_len},), got {feat.shape}"
        )

    def test_no_nans_in_features(self):
        window = _random_window()
        feat = extract_window_features(window, SFREQ)
        assert not np.any(np.isnan(feat)), "Feature vector must not contain NaN"

    def test_no_infs_in_features(self):
        window = _random_window()
        feat = extract_window_features(window, SFREQ)
        assert not np.any(np.isinf(feat)), "Feature vector must not contain Inf"

    def test_constant_channel_no_error(self):
        """A flat EEG channel (e.g., disconnected electrode) should not crash."""
        window = np.zeros((N_CH, WIN), dtype=np.float32)
        feat = extract_window_features(window, SFREQ)
        assert feat is not None
        assert not np.any(np.isnan(feat))


# ─────────────────────────────────────────────────────────────────────────────
# FeatureExtractor — ML mode
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureExtractorML:

    @pytest.fixture
    def extractor(self):
        return FeatureExtractor(mode=FeatureMode.ML, sfreq=SFREQ)

    def test_output_shape(self, extractor):
        X = _random_batch()
        feats = extractor.transform(X)
        expected_cols = N_CH * 16
        assert feats.shape == (8, expected_cols), (
            f"Expected (8, {expected_cols}), got {feats.shape}"
        )

    def test_output_dtype_float32(self, extractor):
        X = _random_batch()
        feats = extractor.transform(X)
        assert feats.dtype == np.float32

    def test_no_nans(self, extractor):
        X = _random_batch()
        feats = extractor.transform(X)
        assert not np.any(np.isnan(feats))

    def test_feature_names_per_channel(self, extractor):
        names = extractor.feature_names
        assert names is not None
        # 16 names per channel
        assert len(names) == 16

    def test_different_windows_give_different_features(self, extractor):
        X1 = _random_batch(n=2)
        X2 = _random_batch(n=2)
        f1 = extractor.transform(X1)
        f2 = extractor.transform(X2)
        # Very unlikely to be identical with random data
        assert not np.allclose(f1, f2)


# ─────────────────────────────────────────────────────────────────────────────
# FeatureExtractor — DL mode
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureExtractorDL:

    @pytest.fixture
    def extractor(self):
        return FeatureExtractor(mode=FeatureMode.DL, sfreq=SFREQ)

    def test_output_shape_preserved(self, extractor):
        X = _random_batch()
        out = extractor.transform(X)
        assert out.shape == X.shape, "DL mode must preserve (N, C, T) shape"

    def test_normalisation_zero_mean(self, extractor):
        X = _random_batch()
        out = extractor.transform(X)
        # After per-window per-channel z-score, mean along T should be ~0
        means = out.mean(axis=-1)    # (N, C)
        assert np.allclose(means, 0.0, atol=1e-5), (
            f"Expected mean≈0, max abs mean = {np.abs(means).max():.6f}"
        )

    def test_normalisation_unit_std(self, extractor):
        X = _random_batch()
        out = extractor.transform(X)
        stds = out.std(axis=-1)      # (N, C)
        assert np.allclose(stds, 1.0, atol=1e-4), (
            f"Expected std≈1, max deviation = {np.abs(stds - 1).max():.6f}"
        )

    def test_constant_channel_handled(self, extractor):
        """Constant channel: std≈0, z-score should not produce NaN/Inf."""
        X = np.zeros((4, N_CH, WIN), dtype=np.float32)
        out = extractor.transform(X)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    def test_output_dtype_float32(self, extractor):
        X = _random_batch()
        out = extractor.transform(X)
        assert out.dtype == np.float32

    def test_feature_names_none_in_dl_mode(self, extractor):
        assert extractor.feature_names is None
