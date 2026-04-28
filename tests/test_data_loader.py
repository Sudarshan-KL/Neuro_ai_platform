"""
tests/test_data_loader.py
--------------------------
Unit tests for EDF loading and seizure annotation parsing.

We use synthetic data (no real EDF files needed for CI) to test parsing logic.
"""

from __future__ import annotations

import textwrap
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Adjust import path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.data_loader.edf_loader import (
    EEGRecord,
    SeizureAnnotationParser,
    SeizureInterval,
)
from ml.preprocessing.windowing import SlidingWindowSegmenter, DatasetBuilder


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_record(
    n_channels: int = 4,
    n_samples: int = 2048,
    sfreq: float = 256.0,
    seizure_ranges: list | None = None,
) -> EEGRecord:
    """Create a synthetic EEGRecord without touching the filesystem."""
    signals = np.random.randn(n_channels, n_samples).astype(np.float32)
    seizures = []
    for s_sec, e_sec in (seizure_ranges or []):
        seizures.append(
            SeizureInterval(
                start_sec=s_sec,
                end_sec=e_sec,
                start_sample=int(s_sec * sfreq),
                end_sample=int(e_sec * sfreq),
            )
        )
    return EEGRecord(
        file_path=Path("synthetic.edf"),
        signals=signals,
        channel_names=[f"CH{i}" for i in range(n_channels)],
        sfreq=sfreq,
        n_samples=n_samples,
        duration_sec=n_samples / sfreq,
        seizures=seizures,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SeizureAnnotationParser
# ─────────────────────────────────────────────────────────────────────────────

class TestSeizureAnnotationParser:

    def _write_annotation(self, text: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".edf.seizures", mode="w", delete=False
        )
        tmp.write(text)
        tmp.flush()
        return Path(tmp.name)

    def test_parse_single_seizure(self):
        content = textwrap.dedent("""\
            File Name: chb01_03.edf
            Number of Seizures in File: 1
            Seizure Start Time: 100 seconds
            Seizure End Time: 140 seconds
        """)
        path = self._write_annotation(content)
        intervals = SeizureAnnotationParser.parse(path, sfreq=256.0, n_samples=50000)
        assert len(intervals) == 1
        assert intervals[0].start_sec == 100
        assert intervals[0].end_sec == 140
        assert intervals[0].start_sample == 25600
        path.unlink()

    def test_parse_multiple_seizures(self):
        content = textwrap.dedent("""\
            Number of Seizures in File: 2
            Seizure Start Time: 50 seconds
            Seizure End Time: 80 seconds
            Seizure Start Time: 200 seconds
            Seizure End Time: 230 seconds
        """)
        path = self._write_annotation(content)
        intervals = SeizureAnnotationParser.parse(path, sfreq=256.0, n_samples=100000)
        assert len(intervals) == 2
        path.unlink()

    def test_missing_annotation_returns_empty(self):
        result = SeizureAnnotationParser.parse(
            Path("/nonexistent/file.edf.seizures"), sfreq=256.0, n_samples=1000
        )
        assert result == []

    def test_invalid_interval_skipped(self):
        """start >= end should produce no intervals."""
        content = textwrap.dedent("""\
            Seizure Start Time: 200 seconds
            Seizure End Time: 100 seconds
        """)
        path = self._write_annotation(content)
        intervals = SeizureAnnotationParser.parse(path, sfreq=256.0, n_samples=100000)
        assert intervals == []
        path.unlink()

    def test_mismatched_counts_returns_empty(self):
        content = textwrap.dedent("""\
            Seizure Start Time: 50 seconds
            Seizure Start Time: 200 seconds
            Seizure End Time: 230 seconds
        """)
        path = self._write_annotation(content)
        intervals = SeizureAnnotationParser.parse(path, sfreq=256.0, n_samples=100000)
        assert intervals == []
        path.unlink()


# ─────────────────────────────────────────────────────────────────────────────
# SlidingWindowSegmenter
# ─────────────────────────────────────────────────────────────────────────────

class TestSlidingWindowSegmenter:

    def test_window_count_no_overlap(self):
        record = make_record(n_samples=1024)
        seg = SlidingWindowSegmenter(window_size=256, stride=256)
        windows = seg.segment(record)
        # 1024 / 256 = exactly 4 non-overlapping windows
        assert len(windows) == 4

    def test_window_count_50pct_overlap(self):
        record = make_record(n_samples=1024)
        seg = SlidingWindowSegmenter(window_size=256, stride=128)
        windows = seg.segment(record)
        # (1024 - 256) / 128 + 1 = 7
        assert len(windows) == 7

    def test_window_shape(self):
        n_ch, ws = 4, 256
        record = make_record(n_channels=n_ch, n_samples=1024)
        seg = SlidingWindowSegmenter(window_size=ws, stride=ws)
        windows = seg.segment(record)
        for w in windows:
            assert w.window.shape == (n_ch, ws)

    def test_no_seizure_all_background(self):
        record = make_record(n_samples=1024, seizure_ranges=[])
        seg = SlidingWindowSegmenter(window_size=256, stride=256)
        windows = seg.segment(record)
        assert all(w.label == 0 for w in windows)

    def test_seizure_window_labeled_correctly(self):
        # Seizure from sample 0 to 255 (first window entirely)
        sfreq = 256.0
        record = make_record(
            n_samples=1024,
            sfreq=sfreq,
            seizure_ranges=[(0.0, 1.0)],  # 0–256 samples
        )
        seg = SlidingWindowSegmenter(window_size=256, stride=256)
        windows = seg.segment(record)
        assert windows[0].label == 1    # first window is seizure
        assert all(w.label == 0 for w in windows[1:])

    def test_overlap_ratio_partial(self):
        sfreq = 256.0
        # Seizure covers second half of first window only
        record = make_record(
            n_samples=1024,
            sfreq=sfreq,
            seizure_ranges=[(0.5, 1.0)],   # samples 128–256
        )
        seg = SlidingWindowSegmenter(
            window_size=256, stride=256, min_overlap_ratio=0.5
        )
        windows = seg.segment(record)
        # Exactly 128/256 = 0.5 overlap → at threshold → label = 1
        assert windows[0].overlap_ratio == pytest.approx(0.5, abs=0.01)
        assert windows[0].label == 1


# ─────────────────────────────────────────────────────────────────────────────
# DatasetBuilder
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetBuilder:

    def test_build_shapes(self):
        records = [make_record(n_samples=1024) for _ in range(3)]
        seg = SlidingWindowSegmenter(window_size=256, stride=256)
        builder = DatasetBuilder(segmenter=seg)
        X, y = builder.build(records)
        # 3 records × 4 windows = 12
        assert X.shape == (12, 4, 256)
        assert y.shape == (12,)
        assert X.dtype == np.float32
        assert set(y.tolist()).issubset({0, 1})

    def test_empty_records_raises(self):
        builder = DatasetBuilder()
        with pytest.raises(ValueError):
            # Empty list → no windows → should raise
            builder.build([])
