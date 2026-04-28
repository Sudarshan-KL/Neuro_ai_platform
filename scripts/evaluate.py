#!/usr/bin/env python3
"""
scripts/evaluate.py
--------------------
Loads a trained model and runs full evaluation on a held-out test set.
Also produces SHAP plots for the RandomForest model.

Usage:
    python scripts/evaluate.py \
        --data_dir data/chbmit \
        --model cnn \
        --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings
from app.core.logging import logger
from ml.data_loader.edf_loader import EDFLoader
from ml.evaluation.evaluator import ModelEvaluator, SHAPExplainer
from ml.features.feature_extractor import FeatureExtractor, FeatureMode
from ml.preprocessing.windowing import DatasetBuilder, SlidingWindowSegmenter
from ml.training.trainer import CNNTrainer, RandomForestTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, default=settings.DATA_DIR)
    p.add_argument("--model",    choices=["rf", "cnn"], default="cnn")
    p.add_argument("--limit",    type=int, default=None)
    p.add_argument("--shap",     action="store_true")
    p.add_argument("--output",   type=Path, default=settings.LOG_DIR / "eval_report.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    loader  = EDFLoader(target_sfreq=settings.SAMPLING_FREQ)
    records = loader.load_directory(args.data_dir, limit=args.limit)

    if not records:
        logger.error("No EDF records found at {}", args.data_dir)
        sys.exit(1)

    segmenter = SlidingWindowSegmenter()
    builder   = DatasetBuilder(segmenter=segmenter)
    X_raw, y  = builder.build(records)

    _, X_test_raw, _, y_test = train_test_split(
        X_raw, y,
        test_size=settings.TEST_SPLIT,
        stratify=y,
        random_state=settings.RANDOM_SEED,
    )

    evaluator = ModelEvaluator()

    if args.model == "rf":
        extractor   = FeatureExtractor(mode=FeatureMode.ML)
        X_test_feat = extractor.transform(X_test_raw)
        model       = RandomForestTrainer.load()
        report      = evaluator.evaluate_sklearn(model, X_test_feat, y_test)

        if args.shap:
            explainer = SHAPExplainer(model)
            n = min(200, len(X_test_feat))
            explainer.explain(X_test_feat[:n], feature_names=extractor.feature_names)

    else:
        extractor  = FeatureExtractor(mode=FeatureMode.DL)
        X_test_dl  = extractor.transform(X_test_raw)
        model      = CNNTrainer.load_model()
        report     = evaluator.evaluate_cnn(model, X_test_dl, y_test)

    # ── Save report ───────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info("Evaluation report saved to {}", args.output)


if __name__ == "__main__":
    main()
