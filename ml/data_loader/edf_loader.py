"""
ml/data_loader/edf_loader.py
----------------------------
STEP 1 — EDF Data Loading
STEP 2 — Seizure Annotation Parsing

Reads raw EDF files via MNE and pairs them with CHB-MIT .seizures annotation
files.  Returns typed dataclasses so every downstream module gets a stable,
self-describing contract.

CHB-MIT .seizures file format (example chb01_03.edf.seizures):
    File Name: chb01_03.edf
    File Start Time: 14:20:26
    File End Time: 15:19:58
    Number of Seizures in File: 1
    Seizure Start Time: 2996 seconds
    Seizure End Time: 3036 seconds
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import mne
import numpy as np

from app.core.logging import logger


# ── Domain types ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SeizureInterval:
    """Inclusive [start_sample, end_sample] of a single seizure event."""
    start_sec: float
    end_sec: float
    start_sample: int
    end_sample: int


@dataclass
class EEGRecord:
    """All data extracted from a single EDF file + its annotation file."""
    file_path: Path
    signals: np.ndarray          # shape: (n_channels, n_samples)
    channel_names: List[str]
    sfreq: float                 # sampling frequency in Hz
    n_samples: int
    duration_sec: float
    seizures: List[SeizureInterval] = field(default_factory=list)

    @property
    def has_seizures(self) -> bool:
        return len(self.seizures) > 0


# ── Seizure annotation parser ──────────────────────────────────────────────────

class SeizureAnnotationParser:
    """
    Parses CHB-MIT .edf.seizures plain-text files.

    The format is intentionally simple but inconsistent across subjects —
    this parser is defensive and handles multiple header variants.
    """

    # Patterns that cover both 'Start Time: N seconds' and 'Start Time: N'
    _START_RE = re.compile(r"seizure start time[:\s]+(\d+)", re.IGNORECASE)
    _END_RE   = re.compile(r"seizure end time[:\s]+(\d+)",   re.IGNORECASE)

    @classmethod
    def parse(
        cls,
        annotation_path: Path,
        sfreq: float,
        n_samples: int,
    ) -> List[SeizureInterval]:
        """
        Parameters
        ----------
        annotation_path : Path to the .edf.seizures file
        sfreq           : Sampling frequency of the paired EDF (Hz)
        n_samples       : Total samples in the EDF (used for bounds checking)

        Returns
        -------
        List of SeizureInterval sorted by start time.
        """
        if not annotation_path.exists():
            logger.debug("No annotation file found at {}", annotation_path)
            return []

        text = annotation_path.read_text(encoding="utf-8", errors="replace")
        starts = [int(m.group(1)) for m in cls._START_RE.finditer(text)]
        ends   = [int(m.group(1)) for m in cls._END_RE.finditer(text)]

        if len(starts) != len(ends):
            logger.warning(
                "Mismatched seizure start/end counts in {} — skipping",
                annotation_path.name,
            )
            return []

        intervals: List[SeizureInterval] = []
        for s_sec, e_sec in zip(starts, ends):
            if s_sec >= e_sec:
                logger.warning(
                    "Invalid seizure interval [{}, {}] in {} — skipping",
                    s_sec, e_sec, annotation_path.name,
                )
                continue

            s_sample = int(s_sec * sfreq)
            e_sample = min(int(e_sec * sfreq), n_samples - 1)
            intervals.append(
                SeizureInterval(
                    start_sec=s_sec,
                    end_sec=e_sec,
                    start_sample=s_sample,
                    end_sample=e_sample,
                )
            )

        logger.info(
            "Parsed {} seizure(s) from {}",
            len(intervals), annotation_path.name,
        )
        return sorted(intervals, key=lambda i: i.start_sample)


# ── EDF loader ────────────────────────────────────────────────────────────────

class EDFLoader:
    """
    Wraps MNE's EDF reader.

    MNE is the gold standard for neurophysiology data in Python.
    We load raw signals as float32 to keep memory manageable for long recordings.
    """

    def __init__(self, target_sfreq: Optional[float] = None) -> None:
        """
        Parameters
        ----------
        target_sfreq : If provided, resample all recordings to this frequency.
                       Set to None to use the file's native sampling rate.
        """
        self.target_sfreq = target_sfreq

    def load(self, edf_path: Path) -> EEGRecord:
        """
        Load a single EDF + its paired .seizures annotation file.

        Parameters
        ----------
        edf_path : Absolute path to the .edf file.

        Returns
        -------
        EEGRecord with signals in µV and seizure intervals in samples.
        """
        edf_path = Path(edf_path)
        if not edf_path.exists():
            raise FileNotFoundError(f"EDF file not found: {edf_path}")

        logger.info("Loading EDF: {}", edf_path.name)

        # Suppress MNE's verbose output — we handle logging ourselves
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

        if self.target_sfreq and raw.info["sfreq"] != self.target_sfreq:
            logger.info(
                "Resampling {} Hz → {} Hz",
                raw.info["sfreq"], self.target_sfreq,
            )
            raw.resample(self.target_sfreq, verbose=False)

        # Extract signals: shape (n_channels, n_times), unit = V → convert to µV
        signals, _ = raw[:]
        signals = (signals * 1e6).astype(np.float32)

        sfreq     = raw.info["sfreq"]
        n_samples = signals.shape[1]

        # Annotation file lives alongside the EDF
        annotation_path = edf_path.with_suffix(".edf.seizures")
        seizures = SeizureAnnotationParser.parse(annotation_path, sfreq, n_samples)

        record = EEGRecord(
            file_path=edf_path,
            signals=signals,
            channel_names=raw.ch_names,
            sfreq=sfreq,
            n_samples=n_samples,
            duration_sec=n_samples / sfreq,
            seizures=seizures,
        )

        logger.info(
            "Loaded {} | channels={} | samples={} | duration={:.1f}s | seizures={}",
            edf_path.name,
            len(record.channel_names),
            n_samples,
            record.duration_sec,
            len(seizures),
        )
        return record

    def load_directory(
        self,
        directory: Path,
        glob_pattern: str = "**/*.edf",
        limit: Optional[int] = None,
    ) -> List[EEGRecord]:
        """
        Recursively load all EDF files under a directory.

        Parameters
        ----------
        directory     : Root directory to search.
        glob_pattern  : Glob to match EDF files.
        limit         : Cap on number of files (useful for quick experiments).

        Returns
        -------
        List of EEGRecord, skipping files that fail to load.
        """
        directory = Path(directory)
        edf_files = sorted(directory.glob(glob_pattern))
        if limit:
            edf_files = edf_files[:limit]

        logger.info(
            "Found {} EDF files under {} (limit={})",
            len(edf_files), directory, limit,
        )

        records: List[EEGRecord] = []
        for path in edf_files:
            try:
                records.append(self.load(path))
            except Exception as exc:
                logger.error("Failed to load {} — {}", path.name, exc)

        logger.info(
            "Successfully loaded {}/{} EDF files",
            len(records), len(edf_files),
        )
        return records
