"""
Preprocessing helpers for image and tabular clinical datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image


def preprocess_image_to_vector(
    image_path: str | Path,
    size: tuple[int, int] = (128, 128),
) -> np.ndarray:
    """Load image, resize, normalize to [0,1], flatten to 1D vector."""
    image = Image.open(image_path).convert("RGB").resize(size)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr.reshape(-1)


def preprocess_image_batch(
    image_paths: Sequence[str | Path],
    size: tuple[int, int] = (128, 128),
) -> np.ndarray:
    """Batch image preprocessing into a 2D array `[n_samples, n_features]`."""
    if not image_paths:
        raise ValueError("No image paths provided for preprocessing.")
    return np.stack([preprocess_image_to_vector(path, size=size) for path in image_paths], axis=0)


def preprocess_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all columns to numeric and impute missing values with median."""
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    return numeric_df.fillna(numeric_df.median(numeric_only=True)).fillna(0.0)

