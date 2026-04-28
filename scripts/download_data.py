#!/usr/bin/env python3
"""
scripts/download_data.py
-------------------------
Downloads the CHB-MIT Scalp EEG dataset from PhysioNet.

Dataset: https://physionet.org/content/chbmit/1.0.0/
License: Open Data Commons Attribution License v1.0

Usage:
    python scripts/download_data.py --subject chb01 --dest data/
    python scripts/download_data.py --all --dest data/       # downloads all 23 subjects
    python scripts/download_data.py --sample --dest data/    # 3 files only (quick start)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging import logger

PHYSIONET_BASE = "https://physionet.org/files/chbmit/1.0.0"

# Subjects available in CHB-MIT (chb01 … chb24, no chb13)
ALL_SUBJECTS = [f"chb{i:02d}" for i in range(1, 25) if i != 13]

# A minimal 3-file sample for quick smoke-tests / CI
SAMPLE_FILES = [
    "chb01/chb01_03.edf",
    "chb01/chb01_03.edf.seizures",
    "chb01/chb01_04.edf",
    "chb01/chb01_04.edf.seizures",
    "chb01/chb01_15.edf",
    "chb01/chb01_15.edf.seizures",
    "chb01/chb01-summary.txt",
]


def _wget(url: str, dest_dir: Path) -> None:
    """Run wget to download a single file."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["wget", "-c", "-q", "--show-progress", "-P", str(dest_dir), url]
    logger.info("Downloading: {}", url)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error("wget failed for: {}", url)


def download_subject(subject: str, dest: Path) -> None:
    """Mirror all EDF + annotation files for one CHB-MIT subject."""
    url = f"{PHYSIONET_BASE}/{subject}/"
    cmd = [
        "wget",
        "-r", "-N", "-c", "-np",   # recursive, newer, continue, no parent
        "--accept", "*.edf,*.edf.seizures,*-summary.txt",
        "-P", str(dest),
        url,
    ]
    logger.info("Downloading subject {} → {}", subject, dest)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error("Download failed for subject: {}", subject)


def download_sample(dest: Path) -> None:
    """Download 3 files from chb01 for quick testing."""
    for rel_path in SAMPLE_FILES:
        url = f"{PHYSIONET_BASE}/{rel_path}"
        subj_dir = dest / Path(rel_path).parent
        _wget(url, subj_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CHB-MIT dataset")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subject", help="Single subject ID, e.g. chb01")
    group.add_argument("--all",    action="store_true", help="Download all subjects")
    group.add_argument("--sample", action="store_true", help="Download 3 sample files from chb01")
    parser.add_argument("--dest", type=Path, default=Path("data"), help="Destination directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Check wget is available
    result = subprocess.run(["wget", "--version"], capture_output=True)
    if result.returncode != 0:
        logger.error("wget not found. Install it: apt-get install wget  or  brew install wget")
        sys.exit(1)

    if args.sample:
        logger.info("Downloading sample files to {}", args.dest)
        download_sample(args.dest)

    elif args.subject:
        if args.subject not in ALL_SUBJECTS:
            logger.error(
                "Unknown subject: {}. Valid: {}",
                args.subject, ALL_SUBJECTS,
            )
            sys.exit(1)
        download_subject(args.subject, args.dest)

    elif args.all:
        logger.info("Downloading all {} subjects. This may take several GB.", len(ALL_SUBJECTS))
        for subj in ALL_SUBJECTS:
            download_subject(subj, args.dest)

    logger.info("Download complete. Data at: {}", args.dest)


if __name__ == "__main__":
    main()
