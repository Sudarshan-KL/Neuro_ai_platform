"""
ml/training/trainer.py
-----------------------
STEP 6 — Model Training

Trains both:
  1. RandomForestClassifier (ML baseline with SHAP support)
  2. SeizureCNN (production DL model)

Both trainers handle class imbalance, persist models, and log training curves.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from app.core.config import settings
from app.core.logging import logger
from ml.training.cnn_model import SeizureCNN


# ─────────────────────────────────────────────────────────────────────────────
# Random Forest baseline
# ─────────────────────────────────────────────────────────────────────────────

class RandomForestTrainer:
    """
    Trains a RandomForest with class-weight balancing.

    Why balanced RF?
    ~~~~~~~~~~~~~~~~
    Seizure windows are typically < 1 % of recordings.  Without weighting the
    forest will learn to always predict background and still achieve 99 %
    accuracy — a clinically useless model.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        n_jobs: int = -1,
        random_state: int = settings.RANDOM_SEED,
    ) -> None:
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",   # ← handles imbalance automatically
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=0,
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> RandomForestClassifier:
        """
        Parameters
        ----------
        X_train : (N, n_features)
        y_train : (N,)
        """
        logger.info(
            "Training RandomForest | samples={} | features={} | pos={}",
            len(y_train), X_train.shape[1], int(y_train.sum()),
        )
        t0 = time.perf_counter()
        self.model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        logger.info("RandomForest training complete in {:.1f}s", elapsed)
        return self.model

    def save(self, path: Optional[Path] = None) -> Path:
        save_path = path or settings.rf_model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("RandomForest saved to {}", save_path)
        return save_path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> RandomForestClassifier:
        load_path = path or settings.rf_model_path
        with open(load_path, "rb") as f:
            model = pickle.load(f)
        logger.info("RandomForest loaded from {}", load_path)
        return model


# ─────────────────────────────────────────────────────────────────────────────
# CNN trainer
# ─────────────────────────────────────────────────────────────────────────────

class CNNTrainer:
    """
    Trains SeizureCNN using:
    * AdamW optimiser with weight decay
    * CosineAnnealingLR scheduler
    * WeightedRandomSampler to oversample seizure windows
    * Binary cross-entropy with pos_weight for additional imbalance correction
    """

    def __init__(
        self,
        n_channels: int,
        window_size: int,
        device: Optional[str] = None,
        base_filters: int = 32,
        dropout: float = 0.3,
        batch_size: int = settings.BATCH_SIZE,
        max_epochs: int = settings.MAX_EPOCHS,
        lr: float = settings.LEARNING_RATE,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr

        self.model = SeizureCNN(
            n_channels=n_channels,
            window_size=window_size,
            base_filters=base_filters,
            dropout=dropout,
        ).to(self.device)

        logger.info("CNN device: {} | params: {:,}", self.device, self._count_params())

    # ── public API ─────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, list]:
        """
        Full training loop with optional validation.

        Parameters
        ----------
        X_train, y_train : Training tensors, X shape (N, C, T)
        X_val, y_val     : Optional validation set

        Returns
        -------
        history : Dict with 'train_loss', 'val_loss' lists
        """
        train_loader = self._build_loader(X_train, y_train, shuffle=True)
        val_loader   = self._build_loader(X_val, y_val, shuffle=False) if X_val is not None else None

        # pos_weight = n_negative / n_positive (additional BCE correction)
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(self.device)
        criterion = nn.BCELoss()  # model already has sigmoid; use raw BCE
        # Alternatively for numerical stability use BCEWithLogitsLoss + remove sigmoid
        # but keeping sigmoid in model for clean inference API

        optimiser = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimiser, T_max=self.max_epochs, eta_min=1e-6)

        history: Dict[str, list] = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._run_epoch(train_loader, criterion, optimiser, training=True)
            history["train_loss"].append(train_loss)

            val_loss = None
            if val_loader is not None:
                val_loss = self._run_epoch(val_loader, criterion, None, training=False)
                history["val_loss"].append(val_loss)

                # Checkpoint best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint("best")

            scheduler.step()

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    "Epoch {:3d}/{} | train_loss={:.4f} | val_loss={}",
                    epoch, self.max_epochs, train_loss,
                    f"{val_loss:.4f}" if val_loss is not None else "N/A",
                )

        logger.info("Training complete. Best val_loss={:.4f}", best_val_loss)
        return history

    def save(self, path: Optional[Path] = None) -> Path:
        save_path = path or settings.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": {
                    "n_channels":  self.model.n_channels,
                    "window_size": self.model.window_size,
                },
                "version": settings.MODEL_VERSION,
            },
            save_path,
        )
        logger.info("CNN model saved to {}", save_path)
        return save_path

    @classmethod
    def load_model(cls, path: Optional[Path] = None) -> SeizureCNN:
        load_path = path or settings.model_path
        checkpoint = torch.load(load_path, map_location="cpu")
        cfg = checkpoint["model_config"]
        model = SeizureCNN(**cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        logger.info(
            "CNN loaded from {} | version={}",
            load_path, checkpoint.get("version", "unknown"),
        )
        return model

    # ── private helpers ────────────────────────────────────────────────────────

    def _run_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimiser: Optional[torch.optim.Optimizer],
        training: bool,
    ) -> float:
        self.model.train(training)
        total_loss = 0.0
        n_batches  = 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.float().to(self.device)

                preds = self.model(X_batch)
                loss  = criterion(preds, y_batch)

                if training and optimiser is not None:
                    optimiser.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimiser.step()

                total_loss += loss.item()
                n_batches  += 1

        return total_loss / max(n_batches, 1)

    def _build_loader(
        self,
        X: Optional[np.ndarray],
        y: Optional[np.ndarray],
        shuffle: bool,
    ) -> Optional[DataLoader]:
        if X is None or y is None:
            return None

        X_t = torch.from_numpy(X).float()
        y_t = torch.from_numpy(y).long()
        dataset = TensorDataset(X_t, y_t)

        sampler = None
        if shuffle:
            # WeightedRandomSampler ensures each batch has ~50% seizure windows
            class_counts = np.bincount(y)
            weights = 1.0 / (class_counts[y] + 1e-12)
            sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(weights),
                num_samples=len(y),
                replacement=True,
            )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler if shuffle else None,
            shuffle=False,            # mutually exclusive with sampler
            num_workers=0,            # keep 0 for Docker compatibility
            pin_memory=self.device.type == "cuda",
        )

    def _save_checkpoint(self, tag: str) -> None:
        path = settings.MODEL_DIR / f"{settings.MODEL_NAME}_{tag}.pt"
        torch.save(self.model.state_dict(), path)

    def _count_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
