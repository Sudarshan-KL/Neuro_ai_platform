"""
Loader utilities for neuro image folders (e.g., brain tumor MRI).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def get_image_samples(data_dir: str | Path) -> List[Tuple[str, str]]:
    """
    Return `(image_path, label)` tuples by scanning class subfolders.
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Neuro image directory not found: {root}")

    samples: List[Tuple[str, str]] = []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for ext in IMAGE_EXTENSIONS:
            for img in class_dir.glob(f"*{ext}"):
                samples.append((str(img), label))
    return samples

