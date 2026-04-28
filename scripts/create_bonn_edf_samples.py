#!/usr/bin/env python3
"""
Create a few single-channel EDF files from the bundled Bonn EEG text dataset.

These samples are compatible with the active Bonn-trained 1-channel model and
are intended for testing the /upload_edf flow end to end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyedflib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BONN_ROOT = PROJECT_ROOT / "Epilepsy-EEG"
OUTPUT_DIR = PROJECT_ROOT / "data" / "bonn-edf-samples"
SFREQ = 173.6

SAMPLE_MAP = {
    "background_A_Z001.edf": BONN_ROOT / "A" / "Z001.txt",
    "background_B_O001.edf": BONN_ROOT / "B" / "O001.txt",
    "seizure_E_S001.edf": BONN_ROOT / "E" / "S001.txt",
}


def write_edf(src_txt: Path, dest_edf: Path) -> None:
    signal = np.loadtxt(src_txt, dtype=np.float64)
    dest_edf.parent.mkdir(parents=True, exist_ok=True)

    channel_info = [{
        "label": "EEG_Fpz-Cz",
        "dimension": "uV",
        "sample_frequency": SFREQ,
        "physical_min": float(signal.min()) - 1.0,
        "physical_max": float(signal.max()) + 1.0,
        "digital_min": -32768,
        "digital_max": 32767,
        "transducer": "synthetic_from_bonn_txt",
        "prefilter": "none",
    }]

    writer = pyedflib.EdfWriter(
        str(dest_edf),
        n_channels=1,
        file_type=pyedflib.FILETYPE_EDFPLUS,
    )
    try:
        writer.setSignalHeaders(channel_info)
        writer.writeSamples([signal])
    finally:
        writer.close()


def main() -> None:
    for output_name, src_txt in SAMPLE_MAP.items():
        if not src_txt.exists():
            raise FileNotFoundError(f"Missing source file: {src_txt}")
        write_edf(src_txt, OUTPUT_DIR / output_name)
        print(f"Created {OUTPUT_DIR / output_name}")


if __name__ == "__main__":
    main()
