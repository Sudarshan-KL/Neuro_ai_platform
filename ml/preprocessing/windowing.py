"""
ml/preprocessing/windowing.py
------------------------------
STEP 3 — Sliding Window Segmentation
STEP 4 — Window Labeling

Converts a continuous EEG recording into fixed-length, overlapping windows
and assigns binary labels (1 = seizure, 0 = background).

Design notes
~~~~~~~~~~~~
* A window is labelled SEIZURE (1) if it overlaps with ANY seizure interval
  by at least `min_overlap_ratio` fraction of its length.  This prevents
  noisy boundary windows from polluting the training signal.
* We deliberately do NOT discard partial-overlap windows — instead we expose
  the overlap ratio so downstream code can apply its own threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from app.core.config import settings
from app.core.logging import logger
from ml.data_loader.edf_loader import EEGRecord, SeizureInterval


# ── Domain types ──────────────────────────────────────────────────────────────

@dataclass
class LabeledWindow:
    """A single EEG window with its binary label and overlap metadata."""
    window: np.ndarray    # shape: (n_channels, window_size)
    label: int            # 0 = background, 1 = seizure
    overlap_ratio: float  # fraction of window that overlaps with a seizure
    start_sample: int     # absolute sample index in the original recording
    end_sample: int


# ── Core windowing logic ───────────────────────────────────────────────────────

class SlidingWindowSegmenter:
    """
    Produces overlapping, fixed-length windows from a multi-channel EEG array.

    Parameters
    ----------
    window_size       : Number of samples per window.
    stride            : Step size between consecutive windows (< window_size
                        gives overlap; = window_size gives non-overlapping).
    min_overlap_ratio : Minimum fraction of window that must overlap a seizure
                        interval to count the window as a positive sample.
                        Default 0.5 avoids mislabelling borderline windows.
    """

    def __init__(
        self,
        window_size: int = settings.WINDOW_SIZE,
        stride: int = settings.WINDOW_STRIDE,
        min_overlap_ratio: float = 0.5,
    ) -> None:
        if stride <= 0:
            raise ValueError("stride must be > 0")
        if window_size <= stride:
            logger.warning(
                "window_size ({}) <= stride ({}) → no overlap between windows",
                window_size, stride,
            )
        self.window_size = window_size
        self.stride = stride
        self.min_overlap_ratio = min_overlap_ratio

    # ── public API ─────────────────────────────────────────────────────────────

    def segment(self, record: EEGRecord) -> List[LabeledWindow]:
        """
        Segment a full EEGRecord into labelled windows.

        Returns
        -------
        List of LabeledWindow objects.
        """
        signals  = record.signals           # (n_channels, n_samples)
        n_ch, n_samp = signals.shape
        seizures = record.seizures

        windows: List[LabeledWindow] = []
        n_windows = 0
        n_seizure_windows = 0

        start = 0
        while start + self.window_size <= n_samp:
            end = start + self.window_size
            window_data = signals[:, start:end]    # (n_channels, window_size)

            overlap_ratio = self._compute_overlap_ratio(start, end, seizures)
            label = int(overlap_ratio >= self.min_overlap_ratio)

            windows.append(
                LabeledWindow(
                    window=window_data,
                    label=label,
                    overlap_ratio=overlap_ratio,
                    start_sample=start,
                    end_sample=end,
                )
            )
            n_windows += 1
            n_seizure_windows += label
            start += self.stride

        logger.info(
            "Segmented {} → {} windows | seizure={} ({:.2f}%) | background={}",
            record.file_path.name,
            n_windows,
            n_seizure_windows,
            100.0 * n_seizure_windows / max(n_windows, 1),
            n_windows - n_seizure_windows,
        )
        return windows

    def segment_raw_array(
        self,
        signals: np.ndarray,
        sfreq: float,
        seizures: Optional[List[SeizureInterval]] = None,
    ) -> List[LabeledWindow]:
        """
        Convenience wrapper for segmenting a raw numpy array (e.g., during
        inference when we don't have a full EEGRecord).
        """
        from ml.data_loader.edf_loader import EEGRecord
        from pathlib import Path

        dummy_record = EEGRecord(
            file_path=Path("stream"),
            signals=signals,
            channel_names=[f"ch{i}" for i in range(signals.shape[0])],
            sfreq=sfreq,
            n_samples=signals.shape[1],
            duration_sec=signals.shape[1] / sfreq,
            seizures=seizures or [],
        )
        return self.segment(dummy_record)

    # ── private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _compute_overlap_ratio(
        win_start: int,
        win_end: int,
        seizures: List[SeizureInterval],
    ) -> float:
        """
        Compute what fraction of [win_start, win_end) overlaps with any seizure.

        We take the UNION of all seizure intervals to avoid double-counting
        when (rare) consecutive seizures overlap.
        """
        win_len = win_end - win_start
        if not seizures or win_len == 0:
            return 0.0

        # Merge overlapping seizure intervals (union)
        sorted_seiz = sorted(seizures, key=lambda s: s.start_sample)
        merged: List[Tuple[int, int]] = []
        for s in sorted_seiz:
            if merged and s.start_sample <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], s.end_sample))
            else:
                merged.append((s.start_sample, s.end_sample))

        overlap_samples = 0
        for s_start, s_end in merged:
            overlap_start = max(win_start, s_start)
            overlap_end   = min(win_end,   s_end + 1)  # +1 → exclusive end
            if overlap_end > overlap_start:
                overlap_samples += overlap_end - overlap_start

        return overlap_samples / win_len


# ── Dataset builder ────────────────────────────────────────────────────────────

class DatasetBuilder:
    """
    Aggregates windows from multiple EEGRecords into numpy arrays ready for
    ML/DL training.
    """

    def __init__(self, segmenter: Optional[SlidingWindowSegmenter] = None) -> None:
        self.segmenter = segmenter or SlidingWindowSegmenter()

    def build(
        self,
        records: List[EEGRecord],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        records : List of EEGRecord.

        Returns
        -------
        X : np.ndarray, shape (n_windows, n_channels, window_size)
        y : np.ndarray, shape (n_windows,), dtype int
        """
        all_windows: List[np.ndarray] = []
        all_labels:  List[int]        = []

        for record in records:
            labeled_windows = self.segmenter.segment(record)
            for lw in labeled_windows:
                all_windows.append(lw.window)
                all_labels.append(lw.label)

        if not all_windows:
            raise ValueError("No windows produced from provided records.")

        X = np.stack(all_windows, axis=0).astype(np.float32)  # (N, C, T)
        y = np.array(all_labels, dtype=np.int64)

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        logger.info(
            "Dataset built | total={} | seizure={} | background={} | imbalance_ratio={:.1f}",
            len(y), n_pos, n_neg,
            n_neg / max(n_pos, 1),
        )
        return X, y
