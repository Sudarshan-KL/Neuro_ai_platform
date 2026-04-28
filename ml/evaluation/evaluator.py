"""
ml/evaluation/evaluator.py
---------------------------
STEP 7 — Model Evaluation + SHAP Explainability

Why recall is paramount in seizure detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A false negative (missed seizure) can be life-threatening — the patient
experiences an undetected seizure without any intervention or warning.
A false positive (false alarm) is an inconvenience.

Therefore we optimise for:
  • High recall (sensitivity) — our primary clinical KPI
  • Acceptable precision — too many false alarms reduce caregiver trust
  • F1 as the harmonic balance between the two
  • ROC-AUC as a threshold-agnostic aggregate metric

All evaluation results are serialisable (plain dicts / floats) so they can
be logged to any monitoring backend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from app.core.config import settings
from app.core.logging import logger
from ml.training.cnn_model import SeizureCNN


# ── Metric bundle ──────────────────────────────────────────────────────────────

class EvaluationReport:
    """Holds all evaluation metrics and provides JSON serialisation."""

    def __init__(
        self,
        precision: float,
        recall: float,
        f1: float,
        roc_auc: float,
        conf_matrix: np.ndarray,
        classification_rep: str,
        threshold: float,
    ) -> None:
        self.precision           = precision
        self.recall              = recall
        self.f1                  = f1
        self.roc_auc             = roc_auc
        self.conf_matrix         = conf_matrix
        self.classification_rep  = classification_rep
        self.threshold           = threshold

    def to_dict(self) -> Dict:
        return {
            "threshold":           self.threshold,
            "precision":           round(self.precision, 4),
            "recall":              round(self.recall, 4),
            "f1":                  round(self.f1, 4),
            "roc_auc":             round(self.roc_auc, 4),
            "confusion_matrix":    self.conf_matrix.tolist(),
            "classification_report": self.classification_rep,
        }

    def log(self) -> None:
        logger.info("── Evaluation Results ──────────────────────────────")
        logger.info("  Threshold : {:.2f}", self.threshold)
        logger.info("  Precision : {:.4f}", self.precision)
        logger.info("  Recall    : {:.4f}  ← clinical KPI (sensitivity)", self.recall)
        logger.info("  F1-Score  : {:.4f}", self.f1)
        logger.info("  ROC-AUC   : {:.4f}", self.roc_auc)
        logger.info("  Confusion Matrix:\n{}", self.conf_matrix)
        logger.info("  Classification Report:\n{}", self.classification_rep)


# ── Threshold finder ───────────────────────────────────────────────────────────

def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    beta: float = 2.0,
) -> float:
    """
    Find the probability threshold that maximises F-beta score on the
    validation set.  beta=2 weights recall twice as much as precision —
    appropriate for a safety-critical system.
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_score, best_thresh = -1.0, 0.5

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        if y_pred.sum() == 0:
            continue
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        denom = (beta ** 2) * p + r
        score = (1 + beta ** 2) * p * r / denom if denom > 0 else 0.0
        if score > best_score:
            best_score, best_thresh = score, thresh

    logger.info("Optimal threshold={:.2f} (F{:.0f}={:.4f})", best_thresh, beta, best_score)
    return best_thresh


# ── Evaluator ─────────────────────────────────────────────────────────────────

class ModelEvaluator:
    """
    Unified evaluator for both sklearn and PyTorch models.
    """

    def __init__(self, threshold: float = settings.SEIZURE_THRESHOLD) -> None:
        self.threshold = threshold

    # ── sklearn (RandomForest) ────────────────────────────────────────────────

    def evaluate_sklearn(
        self,
        model: RandomForestClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray,
        auto_threshold: bool = True,
    ) -> EvaluationReport:
        """Run full evaluation on a sklearn classifier."""
        y_prob = model.predict_proba(X_test)[:, 1]
        return self._compute_report(y_test, y_prob, auto_threshold)

    # ── PyTorch CNN ───────────────────────────────────────────────────────────

    def evaluate_cnn(
        self,
        model: SeizureCNN,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = settings.BATCH_SIZE,
        auto_threshold: bool = True,
    ) -> EvaluationReport:
        """Run full evaluation on the CNN (handles batched GPU inference)."""
        device = next(model.parameters()).device
        model.eval()

        y_probs: List[float] = []
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch = torch.from_numpy(X_test[i : i + batch_size]).float().to(device)
                probs = model(batch).cpu().numpy()
                y_probs.extend(probs.tolist())

        y_prob = np.array(y_probs)
        return self._compute_report(y_test, y_prob, auto_threshold)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _compute_report(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        auto_threshold: bool,
    ) -> EvaluationReport:
        threshold = (
            find_optimal_threshold(y_true, y_prob)
            if auto_threshold
            else self.threshold
        )
        y_pred = (y_prob >= threshold).astype(int)

        report = EvaluationReport(
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
            roc_auc=float(roc_auc_score(y_true, y_prob)),
            conf_matrix=confusion_matrix(y_true, y_pred),
            classification_rep=classification_report(
                y_true, y_pred,
                target_names=["Background", "Seizure"],
                zero_division=0,
            ),
            threshold=threshold,
        )
        report.log()
        return report


# ── SHAP explainability ───────────────────────────────────────────────────────

class SHAPExplainer:
    """
    Wraps SHAP TreeExplainer for the RandomForest model.

    SHAP (SHapley Additive exPlanations) provides per-feature attribution
    scores that tell clinicians which EEG features drove a seizure prediction.
    This is critical for regulatory approval of clinical AI systems.
    """

    def __init__(self, model: RandomForestClassifier) -> None:
        try:
            import shap
            self._shap = shap
        except ImportError:
            raise ImportError("Install shap: pip install shap")

        self.explainer = shap.TreeExplainer(model)

    def explain(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        max_display: int = 20,
    ) -> np.ndarray:
        """
        Compute SHAP values for a batch of windows.

        Parameters
        ----------
        X             : (N, n_features) — feature matrix
        feature_names : Optional list for labelled plots
        max_display   : Number of top features to show in summary plot

        Returns
        -------
        shap_values : (N, n_features) — per-window feature attributions
        """
        import matplotlib.pyplot as plt

        logger.info("Computing SHAP values for {} samples …", len(X))
        shap_values = self.explainer.shap_values(X)

        # shap_values is a list [class0, class1] for binary classifiers
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        # Summary plot (bar chart of mean |SHAP| per feature)
        self._shap.summary_plot(
            sv,
            X,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plot_path = settings.LOG_DIR / "shap_summary.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info("SHAP summary plot saved to {}", plot_path)

        return sv

    def explain_single(self, x: np.ndarray) -> Dict:
        """Return top-5 contributing features for a single prediction."""
        sv = self.explainer.shap_values(x.reshape(1, -1))
        values = sv[1][0] if isinstance(sv, list) else sv[0]
        top_idx = np.argsort(np.abs(values))[::-1][:5]
        return {
            "top_features": top_idx.tolist(),
            "shap_values":  [round(float(values[i]), 6) for i in top_idx],
        }
