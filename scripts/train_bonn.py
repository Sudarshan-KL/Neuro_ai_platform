"""
scripts/train_bonn.py
----------------------
End-to-end training pipeline for the Epilepsy-EEG (Bonn University) dataset.

Usage
-----
    # From the project root:
    python scripts/train_bonn.py

What it does
------------
1. Loads all 500 text files from the Epilepsy-EEG dataset (folders A–E).
2. Applies a sliding-window segmentation (~1-second windows, 50 % overlap).
3. Builds a PyTorch Dataset and DataLoaders with stratified train/test split.
4. Trains a 1-D CNN (SeizureCNN, adapted for 1 channel) on the windowed data.
5. Evaluates and saves the model to saved_models/ with the standard naming so
   the FastAPI inference service picks it up automatically on next startup.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make project root importable regardless of where the script is invoked from
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from app.core.config import settings
from app.core.logging import logger
from ml.data_loader.bonn_loader import BonnDatasetLoader
from ml.training.cnn_model import SeizureCNN


# ──────────────────────────────────────────────────────────────────────────────
# Hyper-parameters (override via env if needed)
# ──────────────────────────────────────────────────────────────────────────────
WINDOW_SIZE   = settings.BONN_WINDOW_SIZE    # ~1 s
WINDOW_STRIDE = settings.BONN_WINDOW_STRIDE  # ~50 % overlap
N_CHANNELS    = settings.BONN_N_CHANNELS     # 1
BATCH_SIZE    = settings.BATCH_SIZE          # 64
MAX_EPOCHS    = settings.MAX_EPOCHS          # 30
LR            = settings.LEARNING_RATE       # 1e-3
TEST_SPLIT    = settings.TEST_SPLIT          # 0.2
SEED          = settings.RANDOM_SEED         # 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class EEGWindowDataset(Dataset):
    """
    PyTorch Dataset wrapping pre-extracted (window, label) pairs.

    Parameters
    ----------
    windows : np.ndarray, shape (N, C, T)
    labels  : np.ndarray, shape (N,)  — int 0/1
    """

    def __init__(self, windows: np.ndarray, labels: np.ndarray) -> None:
        self.windows = torch.from_numpy(windows).float()
        self.labels  = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.windows[idx], self.labels[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Windowing helper
# ──────────────────────────────────────────────────────────────────────────────

def extract_windows(
    signals: np.ndarray,
    label: int,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over a single-channel EEG recording.

    Parameters
    ----------
    signals     : shape (1, n_samples)
    label       : 0 or 1
    window_size : number of samples per window
    stride      : step between windows

    Returns
    -------
    windows : shape (n_windows, 1, window_size)
    labels  : shape (n_windows,)
    """
    n_samples = signals.shape[1]
    windows, labels = [], []

    start = 0
    while start + window_size <= n_samples:
        w = signals[:, start : start + window_size]  # (1, window_size)
        windows.append(w)
        labels.append(label)
        start += stride

    if not windows:
        return np.empty((0, 1, window_size), dtype=np.float32), np.empty(0, dtype=np.int64)

    return np.stack(windows).astype(np.float32), np.array(labels, dtype=np.int64)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE).float()
        optimizer.zero_grad()
        probs = model(x)
        loss = criterion(probs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        preds = (probs >= 0.5).long()
        correct += (preds == y.long()).sum().item()
        total   += len(y)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE).float()
        probs = model(x)
        loss = criterion(probs, y)

        total_loss += loss.item() * len(y)
        preds = (probs >= 0.5).long()
        correct += (preds == y.long()).sum().item()
        total   += len(y)

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y.long().cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
    return avg_loss, accuracy, auc, np.array(all_probs), np.array(all_labels)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    logger.info("=" * 60)
    logger.info("Bonn Epilepsy-EEG Training Pipeline")
    logger.info("=" * 60)
    logger.info("Device      : {}", DEVICE)
    logger.info("Data dir    : {}", settings.BONN_DATA_DIR)
    logger.info("Window size : {} samples (~{:.1f} s)", WINDOW_SIZE, WINDOW_SIZE / settings.BONN_SFREQ)
    logger.info("Stride      : {} samples", WINDOW_STRIDE)
    logger.info("Channels    : {}", N_CHANNELS)

    # ── 1. Load dataset ──────────────────────────────────────────────────────
    loader = BonnDatasetLoader(data_dir=settings.BONN_DATA_DIR)
    records, stats = loader.load_all()
    logger.info("Files per folder: {}", stats)

    # ── 2. Extract windows ───────────────────────────────────────────────────
    all_windows, all_labels = [], []

    for record in records:
        label = 1 if record.has_seizures else 0
        wins, labs = extract_windows(
            record.signals, label, WINDOW_SIZE, WINDOW_STRIDE
        )
        if wins.shape[0] > 0:
            all_windows.append(wins)
            all_labels.append(labs)

    X = np.concatenate(all_windows, axis=0)  # (N, 1, T)
    y = np.concatenate(all_labels,  axis=0)  # (N,)

    n_seizure = int(y.sum())
    n_bg      = int((y == 0).sum())
    logger.info("Total windows : {:,}  |  seizure={:,}  background={:,}", len(y), n_seizure, n_bg)

    # ── 3. Normalise (z-score per window) ───────────────────────────────────
    mean = X.mean(axis=(0, 2), keepdims=True)
    std  = X.std(axis=(0, 2),  keepdims=True) + 1e-8
    X    = (X - mean) / std

    # ── 4. Train / test split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=SEED, stratify=y
    )
    logger.info("Train windows : {:,} | Test windows : {:,}", len(y_train), len(y_test))

    # ── 5. Weighted sampler to fix class imbalance ───────────────────────────
    class_counts = np.bincount(y_train)
    weights      = 1.0 / class_counts[y_train]
    sampler      = WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=len(y_train),
        replacement=True,
    )

    train_ds = EEGWindowDataset(X_train, y_train)
    test_ds  = EEGWindowDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)

    # ── 6. Build model ───────────────────────────────────────────────────────
    model = SeizureCNN(n_channels=N_CHANNELS, window_size=WINDOW_SIZE)
    model = model.to(DEVICE)
    logger.info("Model         : SeizureCNN | params={:,}",
                sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    criterion  = nn.BCELoss()

    # ── 7. Training loop ─────────────────────────────────────────────────────
    best_auc    = 0.0
    best_state  = None

    logger.info("\n{:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
                "Epoch", "TrainLoss", "TrainAcc", "ValLoss", "ValAcc", "ValAUC")
    logger.info("-" * 62)

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, vl_auc, _, _ = eval_epoch(model, test_loader, criterion)
        scheduler.step(vl_loss)

        logger.info("{:>6}  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}",
                    epoch, tr_loss, tr_acc, vl_loss, vl_acc, vl_auc)

        if vl_auc > best_auc:
            best_auc   = vl_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info("  ✓ New best AUC: {:.4f} (epoch {})", best_auc, epoch)

    # ── 8. Final evaluation ──────────────────────────────────────────────────
    logger.info("\nLoading best checkpoint (AUC={:.4f})…", best_auc)
    model.load_state_dict(best_state)

    _, _, _, probs, true_labels = eval_epoch(model, test_loader, criterion)
    preds = (probs >= 0.5).astype(int)

    logger.info("\nClassification Report:\n{}",
                classification_report(true_labels, preds,
                                      target_names=["Background", "Seizure"]))
    logger.info("Confusion Matrix:\n{}", confusion_matrix(true_labels, preds))
    logger.info("ROC-AUC: {:.4f}", roc_auc_score(true_labels, probs))

    # ── 9. Save model ────────────────────────────────────────────────────────
    settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = settings.MODEL_DIR / f"{settings.MODEL_NAME}_{settings.MODEL_VERSION}.pt"

    torch.save(
        {
            "model_state_dict": best_state,
            "model_config": {
                "n_channels":    N_CHANNELS,
                "window_size":   WINDOW_SIZE,
            },
            "training_config": {
                "dataset":       "bonn_epilepsy_eeg",
                "sfreq":         settings.BONN_SFREQ,
                "window_size":   WINDOW_SIZE,
                "window_stride": WINDOW_STRIDE,
                "n_channels":    N_CHANNELS,
                "best_auc":      best_auc,
                "epochs":        MAX_EPOCHS,
                "seed":          SEED,
            },
            "norm_stats": {
                "mean": mean.tolist(),
                "std":  std.tolist(),
            },
        },
        save_path,
    )
    logger.info("\nModel saved → {}", save_path)
    logger.info("Done. Best validation AUC = {:.4f}", best_auc)


if __name__ == "__main__":
    main()
