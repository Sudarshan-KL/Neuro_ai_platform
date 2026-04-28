from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.dummy import DummyClassifier

from app.core.config import settings
from ml.data_loader.alzheimers_loader import get_alzheimers_image_paths
from ml.data_loader.neuro_loader import get_image_samples
from ml.data_loader.parkinsons_loader import (
    load_parkinsons_csv,
    split_parkinsons_features_target,
)
from ml.preprocessing.multimodal import (
    preprocess_image_batch,
    preprocess_image_to_vector,
    preprocess_tabular_features,
)


class DisorderModelService:
    """Train/load lightweight disorder-specific models for API inference."""

    def __init__(self) -> None:
        self.model_dir: Path = settings.MODEL_DIR
        self.alz_path = self.model_dir / "alzheimers_image_classifier.pkl"
        self.parkinson_path = self.model_dir / "parkinsons_tabular_classifier.pkl"
        self.neuro_path = self.model_dir / "neuro_image_classifier.pkl"
        self._results: List[Dict[str, Any]] = []

    def _record(self, disorder: str, prediction: str, confidence: float, extra: Dict[str, Any] | None = None) -> None:
        self._results.append(
            {
                "timestamp": time.time(),
                "disorder": disorder,
                "prediction": prediction,
                "confidence": round(float(confidence), 4),
                "extra": extra or {},
            }
        )
        self._results = self._results[-200:]

    def _train_image_classifier(self, samples: List[tuple[str, str]], save_path: Path) -> Dict[str, Any]:
        # Keep startup training lightweight for local demo environments.
        max_samples = 120
        if len(samples) > max_samples:
            samples = samples[:max_samples]
        image_paths = [s[0] for s in samples]
        labels = [s[1] for s in samples]
        X = preprocess_image_batch(image_paths, size=(64, 64))
        le = LabelEncoder()
        y = le.fit_transform(labels)
        classifier = (
            LogisticRegression(max_iter=300)
            if len(set(y)) > 1
            else DummyClassifier(strategy="most_frequent")
        )
        model = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", classifier)])
        model.fit(X, y)
        artifact = {
            "model": model,
            "label_encoder": le,
            "image_size": (64, 64),
            "n_samples": len(samples),
            "classes": list(le.classes_),
        }
        joblib.dump(artifact, save_path)
        return artifact

    def _train_parkinsons_classifier(self) -> Dict[str, Any]:
        df = load_parkinsons_csv(settings.DATA_DIR / "Parkinsson disease.csv")
        X_raw, y = split_parkinsons_features_target(df)
        X = preprocess_tabular_features(X_raw)
        classifier = (
            LogisticRegression(max_iter=600)
            if len(set(y)) > 1
            else DummyClassifier(strategy="most_frequent")
        )
        model = Pipeline([("scaler", StandardScaler()), ("clf", classifier)])
        model.fit(X, y)
        artifact = {
            "model": model,
            "feature_columns": list(X.columns),
            "target": "status",
            "n_samples": int(len(X)),
        }
        joblib.dump(artifact, self.parkinson_path)
        return artifact

    def _train_alzheimers_classifier(self) -> Dict[str, Any]:
        samples = get_alzheimers_image_paths(settings.DATA_DIR / "alzheimers")
        if not samples:
            raise ValueError("No Alzheimer image samples found in data/alzheimers.")
        return self._train_image_classifier(samples, self.alz_path)

    def _train_neuro_classifier(self) -> Dict[str, Any]:
        neuro_root = settings.DATA_DIR / "neuro"
        if (neuro_root / "brain_tumor_dataset").exists():
            neuro_root = neuro_root / "brain_tumor_dataset"
        samples = get_image_samples(neuro_root)
        if not samples:
            raise ValueError("No neuro image samples found in data/neuro.")
        return self._train_image_classifier(samples, self.neuro_path)

    def ensure_artifacts(self) -> None:
        if not self.alz_path.exists():
            self._train_alzheimers_classifier()
        if not self.parkinson_path.exists():
            self._train_parkinsons_classifier()
        if not self.neuro_path.exists():
            self._train_neuro_classifier()

    def _load(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            self.ensure_artifacts()
        return joblib.load(path)

    def predict_alzheimers(self, image_path: str | Path) -> Dict[str, Any]:
        artifact = self._load(self.alz_path)
        x = preprocess_image_to_vector(image_path, artifact.get("image_size", (128, 128))).reshape(1, -1)
        probs = artifact["model"].predict_proba(x)[0]
        idx = int(np.argmax(probs))
        label = artifact["label_encoder"].inverse_transform([idx])[0]
        conf = float(probs[idx])
        self._record("alzheimers", label, conf)
        return {"prediction": label, "confidence": conf, "classes": artifact.get("classes", [])}

    def predict_neuro(self, image_path: str | Path) -> Dict[str, Any]:
        artifact = self._load(self.neuro_path)
        x = preprocess_image_to_vector(image_path, artifact.get("image_size", (128, 128))).reshape(1, -1)
        probs = artifact["model"].predict_proba(x)[0]
        idx = int(np.argmax(probs))
        label = artifact["label_encoder"].inverse_transform([idx])[0]
        conf = float(probs[idx])
        self._record("neuro", label, conf)
        return {"prediction": label, "confidence": conf, "classes": artifact.get("classes", [])}

    def predict_parkinsons(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        artifact = self._load(self.parkinson_path)
        feature_cols = artifact["feature_columns"]
        row = {k: payload.get(k, 0.0) for k in feature_cols}
        df = preprocess_tabular_features(pd.DataFrame([row]))[feature_cols]
        probs = artifact["model"].predict_proba(df)[0]
        pred = int(artifact["model"].predict(df)[0])
        conf = float(np.max(probs))
        label = "parkinsons_positive" if pred == 1 else "parkinsons_negative"
        self._record("parkinsons", label, conf)
        return {"prediction": label, "confidence": conf, "target_value": pred}

    def model_info(self) -> Dict[str, Any]:
        self.ensure_artifacts()
        artifacts = {
            "alzheimers": self._load(self.alz_path),
            "parkinsons": self._load(self.parkinson_path),
            "neuro": self._load(self.neuro_path),
        }
        return {
            "alzheimers": {
                "artifact": str(self.alz_path),
                "classes": artifacts["alzheimers"].get("classes", []),
                "n_samples": artifacts["alzheimers"].get("n_samples"),
            },
            "parkinsons": {
                "artifact": str(self.parkinson_path),
                "target": artifacts["parkinsons"].get("target"),
                "feature_count": len(artifacts["parkinsons"].get("feature_columns", [])),
                "n_samples": artifacts["parkinsons"].get("n_samples"),
            },
            "neuro": {
                "artifact": str(self.neuro_path),
                "classes": artifacts["neuro"].get("classes", []),
                "n_samples": artifacts["neuro"].get("n_samples"),
            },
        }

    def dataset_info(self) -> Dict[str, Any]:
        parkinson_df = load_parkinsons_csv(settings.DATA_DIR / "Parkinsson disease.csv")
        alz_samples = get_alzheimers_image_paths(settings.DATA_DIR / "alzheimers")
        neuro_root = settings.DATA_DIR / "neuro"
        if (neuro_root / "brain_tumor_dataset").exists():
            neuro_root = neuro_root / "brain_tumor_dataset"
        neuro_samples = get_image_samples(neuro_root)

        return {
            "alzheimers": {
                "path": str(settings.DATA_DIR / "alzheimers"),
                "samples": len(alz_samples),
            },
            "parkinsons": {
                "path": str(settings.DATA_DIR / "Parkinsson disease.csv"),
                "rows": len(parkinson_df),
                "columns": list(parkinson_df.columns),
            },
            "neuro": {"path": str(neuro_root), "samples": len(neuro_samples)},
        }

    @property
    def results(self) -> List[Dict[str, Any]]:
        return list(self._results)

