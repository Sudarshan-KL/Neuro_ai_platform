"""
Data loader for Parkinson's Disease tabular voice-feature dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def load_parkinsons_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load Parkinson's CSV file into a DataFrame."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Parkinson dataset not found: {path}")
    return pd.read_csv(path)


def split_parkinsons_features_target(
    df: pd.DataFrame,
    target_col: str = "status",
    drop_columns: tuple[str, ...] = ("name",),
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split Parkinson's DataFrame into numeric features and binary target.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col, *[c for c in drop_columns if c in df.columns]])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X, y
