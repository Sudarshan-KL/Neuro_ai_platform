"""
ml/data_loader/bonn_loader.py
------------------------------
Data loader for the Kaggle Epilepsy-EEG (Bonn University) dataset.

Dataset description
~~~~~~~~~~~~~~~~~~~
The Bonn dataset consists of 5 folders (A–E), each containing 100 single-
channel EEG recordings stored as plain text files (one sample per line,
4097 samples per file, sampled at 173.6 Hz).

Folder — Class mapping:
    A (Z-prefix) : Healthy volunteers, eyes open       → Background (0)
    B (O-prefix) : Healthy volunteers, eyes closed      → Background (0)
    C (N-prefix) : Epileptic patients, hippocampal      → Background (0)
    D (F-prefix) : Epileptic patients, epileptogenic    → Background (0)
    E (S-prefix) : Epileptic patients, seizure activity → Seizure    (1)

Reference:
    Andrzejak RG et al., "Indications of nonlinear deterministic and
    finite-dimensional structures in time series of brain electrical
    activity", Physical Review E, 2001.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.core.logging import logger
from ml.data_loader.edf_loader import EEGRecord, SeizureInterval


# Bonn dataset constants
BONN_SFREQ = 173.6           # Sampling frequency in Hz
BONN_SAMPLES_PER_FILE = 4097 # Each file has exactly 4097 samples
BONN_N_CHANNELS = 1          # Single-channel recordings

# Folder → label mapping
SEIZURE_FOLDERS = {"E"}       # Only folder E contains seizure recordings
BACKGROUND_FOLDERS = {"A", "B", "C", "D"}

# All valid folders
ALL_FOLDERS = SEIZURE_FOLDERS | BACKGROUND_FOLDERS


@dataclass
class BonnFileRecord:
    """Metadata for a single Bonn dataset text file."""
    file_path: Path
    folder: str          # A, B, C, D, or E
    label: int           # 0 = background, 1 = seizure
    signals: np.ndarray  # shape: (1, n_samples) — single channel


class BonnDatasetLoader:
    """
    Loads the Epilepsy-EEG (Bonn) dataset from disk.

    Reads plain-text files from folders A–E and returns EEGRecord objects
    compatible with the existing windowing pipeline.

    Parameters
    ----------
    data_dir : Path to the root Epilepsy-EEG directory containing A/ B/ C/ D/ E/ subfolders.
    folders  : Which folders to include. Default: all five.
    """

    def __init__(
        self,
        data_dir: Path,
        folders: Optional[List[str]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.folders = folders or sorted(ALL_FOLDERS)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Bonn dataset directory not found: {self.data_dir}")

        for folder in self.folders:
            if folder not in ALL_FOLDERS:
                raise ValueError(f"Invalid folder '{folder}'. Must be one of {ALL_FOLDERS}")

    def _read_txt_file(self, file_path: Path) -> np.ndarray:
        """
        Read a single Bonn text file.

        Each file contains one integer sample value per line (4097 lines).
        Returns a 1-D numpy array of float32 values.
        """
        try:
            values = np.loadtxt(str(file_path), dtype=np.float32)
            return values
        except Exception as exc:
            logger.error("Failed to read Bonn file {} — {}", file_path.name, exc)
            raise

    def load_file(self, file_path: Path, folder: str) -> EEGRecord:
        """
        Load a single Bonn text file and wrap it as an EEGRecord.

        Parameters
        ----------
        file_path : Path to the .txt file.
        folder    : Parent folder name (A/B/C/D/E) — used for labeling.

        Returns
        -------
        EEGRecord with shape (1, n_samples) signals.
        """
        raw_signal = self._read_txt_file(file_path)  # shape: (n_samples,)
        signals = raw_signal.reshape(1, -1)           # shape: (1, n_samples)
        n_samples = signals.shape[1]

        # Build seizure intervals: for folder E the entire file is a seizure
        seizures: List[SeizureInterval] = []
        if folder in SEIZURE_FOLDERS:
            seizures.append(
                SeizureInterval(
                    start_sec=0.0,
                    end_sec=n_samples / BONN_SFREQ,
                    start_sample=0,
                    end_sample=n_samples - 1,
                )
            )

        return EEGRecord(
            file_path=file_path,
            signals=signals,
            channel_names=[f"EEG_{folder}"],
            sfreq=BONN_SFREQ,
            n_samples=n_samples,
            duration_sec=n_samples / BONN_SFREQ,
            seizures=seizures,
        )

    def load_all(
        self,
        limit_per_folder: Optional[int] = None,
    ) -> Tuple[List[EEGRecord], Dict[str, int]]:
        """
        Load all files from the configured folders.

        Parameters
        ----------
        limit_per_folder : If set, only load this many files per folder
                           (for quick experiments).

        Returns
        -------
        records : List of EEGRecord objects.
        stats   : Dict with per-folder file counts.
        """
        records: List[EEGRecord] = []
        stats: Dict[str, int] = {}

        for folder in self.folders:
            folder_path = self.data_dir / folder
            if not folder_path.exists():
                logger.warning("Folder {} not found at {}, skipping", folder, folder_path)
                continue

            # Collect .txt and .TXT files (folder C uses uppercase .TXT)
            txt_files = sorted(
                list(folder_path.glob("*.txt")) + list(folder_path.glob("*.TXT"))
            )
            # Deduplicate (on case-insensitive filesystems both globs may match the same files)
            seen = set()
            unique_files = []
            for f in txt_files:
                key = f.name.lower()
                if key not in seen:
                    seen.add(key)
                    unique_files.append(f)
            txt_files = unique_files

            if limit_per_folder:
                txt_files = txt_files[:limit_per_folder]

            stats[folder] = len(txt_files)
            label_name = "SEIZURE" if folder in SEIZURE_FOLDERS else "BACKGROUND"

            for file_path in txt_files:
                try:
                    record = self.load_file(file_path, folder)
                    records.append(record)
                except Exception as exc:
                    logger.error("Skipping {} — {}", file_path.name, exc)

            logger.info(
                "Folder {} | {} files | label={}",
                folder, len(txt_files), label_name,
            )

        total_seizure = sum(1 for r in records if r.has_seizures)
        total_background = len(records) - total_seizure
        logger.info(
            "Bonn dataset loaded | total={} | seizure={} | background={} | folders={}",
            len(records), total_seizure, total_background, stats,
        )

        return records, stats
