"""
ml/training/cnn_model.py
------------------------
1D Convolutional Neural Network for EEG seizure detection.

Architecture: multi-scale temporal convolutions → squeeze-excitation → 
              batch norm → global average pooling → binary classifier.

Design decisions
~~~~~~~~~~~~~~~~
* Multi-scale kernels (7, 15, 31) capture fast spikes AND slow rhythmic 
  activity characteristic of seizure onset.
* Squeeze-and-Excitation (SE) block re-weights channel importance dynamically.
* Dropout after each block prevents co-adaptation on heavily imbalanced data.
* Sigmoid output → probability score used by the alert threshold logic.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ────────────────────────────────────────────────────────────

class SqueezeExcitation(nn.Module):
    """
    Channel-wise attention from 'Squeeze-and-Excitation Networks' (Hu et al.).
    Learns to emphasise informative frequency channels (in our case EEG electrode
    channels after the first projection conv).
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        scale = self.se(x).unsqueeze(-1)   # (B, C, 1)
        return x * scale


class ConvBlock(nn.Module):
    """Conv1d → BN → ReLU → Dropout building block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_ch, out_ch, kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiScaleBlock(nn.Module):
    """
    Applies three parallel conv branches with different kernel sizes,
    concatenates their outputs, then projects back to `out_ch`.
    Inspired by Inception architecture but adapted for 1-D time series.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernels: Tuple[int, ...] = (7, 15, 31),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        branch_ch = out_ch // len(kernels)

        self.branches = nn.ModuleList([
            ConvBlock(in_ch, branch_ch, k, dropout=dropout)
            for k in kernels
        ])
        # Project concatenated branches back to out_ch
        self.proj = nn.Sequential(
            nn.Conv1d(branch_ch * len(kernels), out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.se = SqueezeExcitation(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [branch(x) for branch in self.branches]
        # Pad to same length before concatenating (parallel convs may differ by ±1)
        min_len = min(o.size(-1) for o in outs)
        outs = [o[..., :min_len] for o in outs]
        x = torch.cat(outs, dim=1)
        x = self.proj(x)
        return self.se(x)


# ── Main model ────────────────────────────────────────────────────────────────

class SeizureCNN(nn.Module):
    """
    End-to-end 1-D CNN for EEG seizure detection.

    Input : (batch, n_channels, window_size)
    Output: (batch,)  — seizure probability in [0, 1]
    """

    def __init__(
        self,
        n_channels: int,
        window_size: int,
        base_filters: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.n_channels  = n_channels
        self.window_size = window_size

        # ── Stem: project raw EEG channels to feature maps ───────────────────
        self.stem = ConvBlock(n_channels, base_filters, kernel_size=7, dropout=dropout)

        # ── Multi-scale blocks ────────────────────────────────────────────────
        self.block1 = MultiScaleBlock(base_filters,     base_filters * 2, dropout=dropout)
        self.pool1  = nn.MaxPool1d(kernel_size=2, stride=2)

        self.block2 = MultiScaleBlock(base_filters * 2, base_filters * 4, dropout=dropout)
        self.pool2  = nn.MaxPool1d(kernel_size=2, stride=2)

        self.block3 = MultiScaleBlock(base_filters * 4, base_filters * 8, dropout=dropout)

        # ── Global average pool → dense head ─────────────────────────────────
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, T) — batch of EEG windows

        Returns
        -------
        probs : (B,) — seizure probability per window
        """
        x = self.stem(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)
        x = self.gap(x)
        return self.classifier(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward() — matches sklearn API convention."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)
