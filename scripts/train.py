#!/usr/bin/env python3
"""
scripts/train.py
----------------
End-to-end training pipeline.

Usage:
    python scripts/train.py \
        --data_dir data/chbmit \
        --model cnn \
        --limit 10

Steps:
  1. Load EDF files from data_dir
  2. Segment into windows
  3. Extract features (Mode A or B based on --model flag)
  4. Train/test split with stratification
  5. Train model (RF or CNN)
  6. Evaluate
  7. Save artefacts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings
from app.core.logging import logger
from ml.data_loader.edf_loader import EDFLoader
from ml.evaluation.evaluator import ModelEvaluator, SHAPExplainer
from ml.features.feature_extractor import FeatureExtractor, FeatureMode
from ml.preprocessing.windowing import DatasetBuilder, SlidingWindowSegmenter
from ml.training.trainer import CNNTrainer, RandomForestTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train seizure detection model")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=settings.DATA_DIR,
        help="Root directory containing EDF files",
    )
    parser.add_argument(
        "--model",
        choices=["rf", "cnn", "both"],
        default="cnn",
        help="Model to train",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of EDF files (for quick experiments)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=settings.WINDOW_SIZE,
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=settings.WINDOW_STRIDE,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=settings.MAX_EPOCHS,
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Run SHAP explainability after RF training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings.ensure_dirs()

    # ── 1. Data loading ───────────────────────────────────────────────────────
    logger.info("=== STEP 1: Data Loading ===")
    loader  = EDFLoader(target_sfreq=settings.SAMPLING_FREQ)
    records = loader.load_directory(args.data_dir, limit=args.limit)

    if not records:
        logger.error("No EDF records loaded from {}. Exiting.", args.data_dir)
        sys.exit(1)

    # ── 2. Windowing + labeling ───────────────────────────────────────────────
    logger.info("=== STEP 2-3: Windowing & Labeling ===")
    segmenter = SlidingWindowSegmenter(
        window_size=args.window_size,
        stride=args.stride,
    )
    builder = DatasetBuilder(segmenter=segmenter)
    X_raw, y = builder.build(records)
    logger.info("Raw dataset: X={} y={} pos={}", X_raw.shape, y.shape, int(y.sum()))

    # ── 3. Train/test split ───────────────────────────────────────────────────
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y,
        test_size=settings.TEST_SPLIT,
        stratify=y,
        random_state=settings.RANDOM_SEED,
    )
    logger.info(
        "Train={} | Test={} | train_pos={} | test_pos={}",
        len(y_train), len(y_test), int(y_train.sum()), int(y_test.sum()),
    )

    evaluator = ModelEvaluator()

    # ── 4a. Random Forest ─────────────────────────────────────────────────────
    if args.model in ("rf", "both"):
        logger.info("=== STEP 4a: Random Forest ===")
        extractor = FeatureExtractor(mode=FeatureMode.ML)
        X_train_feat = extractor.transform(X_train_raw)
        X_test_feat  = extractor.transform(X_test_raw)

        rf_trainer = RandomForestTrainer()
        rf_model   = rf_trainer.train(X_train_feat, y_train)

        logger.info("=== Evaluating RandomForest ===")
        rf_report = evaluator.evaluate_sklearn(rf_model, X_test_feat, y_test)
        rf_trainer.save()

        if args.shap:
            logger.info("=== SHAP Explainability ===")
            try:
                explainer = SHAPExplainer(rf_model)
                # Use a sample of test data for speed
                n_shap = min(500, len(X_test_feat))
                explainer.explain(
                    X_test_feat[:n_shap],
                    feature_names=extractor.feature_names,
                )
            except Exception as e:
                logger.warning("SHAP failed: {}", e)

    # ── 4b. CNN ───────────────────────────────────────────────────────────────
    if args.model in ("cnn", "both"):
        logger.info("=== STEP 4b: CNN Training ===")
        extractor = FeatureExtractor(mode=FeatureMode.DL)
        X_train_dl = extractor.transform(X_train_raw)
        X_test_dl  = extractor.transform(X_test_raw)

        # Further split training into train / val
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_dl, y_train,
            test_size=0.1,
            stratify=y_train,
            random_state=settings.RANDOM_SEED,
        )

        n_channels, window_size = X_tr.shape[1], X_tr.shape[2]
        cnn_trainer = CNNTrainer(
            n_channels=n_channels,
            window_size=window_size,
            max_epochs=args.epochs,
        )
        history = cnn_trainer.train(X_tr, y_tr, X_val, y_val)
        cnn_trainer.save()

        logger.info("=== Evaluating CNN ===")
        cnn_report = evaluator.evaluate_cnn(
            cnn_trainer.model, X_test_dl, y_test
        )

    logger.info("=== Training pipeline complete. ===")
    logger.info(
        "Models saved to: {}",
        settings.MODEL_DIR,
    )


if __name__ == "__main__":
    main()
