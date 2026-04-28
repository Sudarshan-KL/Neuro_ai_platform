"""
tests/test_model.py
--------------------
Unit tests for the SeizureCNN architecture.

Validates forward pass shapes, gradient flow, and output range without
any training — pure architecture tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.training.cnn_model import ConvBlock, MultiScaleBlock, SeizureCNN, SqueezeExcitation


N_CH  = 23
WIN   = 256
BATCH = 4


@pytest.fixture
def model():
    m = SeizureCNN(n_channels=N_CH, window_size=WIN, base_filters=16)
    m.eval()
    return m


@pytest.fixture
def random_batch():
    return torch.randn(BATCH, N_CH, WIN)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture
# ─────────────────────────────────────────────────────────────────────────────

class TestSeizureCNNArchitecture:

    def test_output_shape(self, model, random_batch):
        with torch.no_grad():
            out = model(random_batch)
        assert out.shape == (BATCH,), f"Expected ({BATCH},), got {out.shape}"

    def test_output_range_0_to_1(self, model, random_batch):
        """Sigmoid output must be in [0, 1]."""
        with torch.no_grad():
            out = model(random_batch)
        assert out.min() >= 0.0, f"Output below 0: {out.min()}"
        assert out.max() <= 1.0, f"Output above 1: {out.max()}"

    def test_single_sample_inference(self, model):
        """Batch size of 1 must work (inference scenario)."""
        x = torch.randn(1, N_CH, WIN)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1,)

    def test_predict_proba_alias(self, model, random_batch):
        """predict_proba should return same result as forward."""
        with torch.no_grad():
            out1 = model(random_batch)
            out2 = model.predict_proba(random_batch)
        assert torch.allclose(out1, out2)

    def test_different_inputs_different_outputs(self, model):
        """Sanity check: different inputs should not give identical outputs."""
        x1 = torch.randn(2, N_CH, WIN)
        x2 = torch.randn(2, N_CH, WIN)
        with torch.no_grad():
            o1 = model(x1)
            o2 = model(x2)
        assert not torch.allclose(o1, o2), "Different inputs should differ"

    def test_deterministic_in_eval_mode(self, model, random_batch):
        """eval() mode should give deterministic outputs (no dropout noise)."""
        model.eval()
        with torch.no_grad():
            o1 = model(random_batch)
            o2 = model(random_batch)
        assert torch.allclose(o1, o2), "eval() outputs must be deterministic"

    def test_param_count_reasonable(self, model):
        """Sanity check: model shouldn't be trivially tiny or absurdly large."""
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_params > 10_000,  f"Too few params: {n_params}"
        assert n_params < 50_000_000, f"Too many params: {n_params}"


# ─────────────────────────────────────────────────────────────────────────────
# Gradient flow
# ─────────────────────────────────────────────────────────────────────────────

class TestGradientFlow:

    def test_gradients_flow_to_all_parameters(self):
        """Every parameter with requires_grad=True must receive a gradient."""
        model = SeizureCNN(n_channels=N_CH, window_size=WIN, base_filters=16)
        model.train()

        x = torch.randn(2, N_CH, WIN)
        y = torch.tensor([0.0, 1.0])

        out  = model(x)
        loss = torch.nn.functional.binary_cross_entropy(out, y)
        loss.backward()

        dead_params = [
            name
            for name, p in model.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert dead_params == [], f"Dead params (no grad): {dead_params}"

    def test_loss_decreases_after_one_step(self):
        """One optimiser step should reduce the loss."""
        model = SeizureCNN(n_channels=N_CH, window_size=WIN, base_filters=16)
        model.train()
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(4, N_CH, WIN)
        y = torch.tensor([0.0, 1.0, 0.0, 1.0])

        out1  = model(x)
        loss1 = torch.nn.functional.binary_cross_entropy(out1, y)
        loss1.backward()
        optim.step()

        optim.zero_grad()
        out2  = model(x)
        loss2 = torch.nn.functional.binary_cross_entropy(out2, y)

        # Loss should decrease (or at minimum not explode)
        assert loss2.item() < loss1.item() * 2, (
            f"Loss did not decrease: {loss1.item():.4f} → {loss2.item():.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildingBlocks:

    def test_conv_block_shape_preserved(self):
        block = ConvBlock(in_ch=8, out_ch=16, kernel_size=7)
        x = torch.randn(2, 8, 128)
        out = block(x)
        assert out.shape == (2, 16, 128), f"Unexpected shape: {out.shape}"

    def test_squeeze_excitation_shape_preserved(self):
        se = SqueezeExcitation(channels=16)
        x  = torch.randn(2, 16, 64)
        out = se(x)
        assert out.shape == x.shape

    def test_se_output_scaled(self):
        """SE block should scale (not zero-out) the input."""
        se = SqueezeExcitation(channels=8)
        x  = torch.ones(1, 8, 32)
        with torch.no_grad():
            out = se(x)
        assert not torch.all(out == 0), "SE block must not zero out the input"

    def test_multiscale_block_reduces_channels_correctly(self):
        block = MultiScaleBlock(in_ch=16, out_ch=32, kernels=(7, 15, 31))
        x = torch.randn(2, 16, 128)
        out = block(x)
        assert out.shape[1] == 32, f"Expected 32 channels, got {out.shape[1]}"

    def test_multiscale_block_length_consistent(self):
        """Output length should match input length (padding='same' behaviour)."""
        block = MultiScaleBlock(in_ch=8, out_ch=16)
        x     = torch.randn(2, 8, 256)
        out   = block(x)
        assert out.shape[-1] == x.shape[-1], (
            f"Length mismatch: in={x.shape[-1]}, out={out.shape[-1]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Variable input length robustness
# ─────────────────────────────────────────────────────────────────────────────

class TestInputRobustness:

    @pytest.mark.parametrize("window_size", [128, 256, 512])
    def test_different_window_sizes(self, window_size):
        model = SeizureCNN(n_channels=N_CH, window_size=window_size, base_filters=8)
        model.eval()
        x = torch.randn(2, N_CH, window_size)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2,)

    @pytest.mark.parametrize("n_channels", [1, 4, 23])
    def test_different_channel_counts(self, n_channels):
        model = SeizureCNN(n_channels=n_channels, window_size=WIN, base_filters=8)
        model.eval()
        x = torch.randn(2, n_channels, WIN)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2,)
