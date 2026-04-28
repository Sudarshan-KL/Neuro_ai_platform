#!/usr/bin/env python3
"""
scripts/stream_demo.py
-----------------------
Demonstrates real-time seizure detection by replaying an EDF file
through the InferenceService's streaming buffer.

Requires a trained CNN model at saved_models/.

Usage:
    python scripts/stream_demo.py \
        --edf data/chbmit/chb01/chb01_03.edf \
        --delay 0.05
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings
from app.core.logging import logger
from app.models.model_registry import registry
from app.services.inference import InferenceService, ModelWrapper, simulate_realtime_stream
from ml.data_loader.edf_loader import EDFLoader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time EEG seizure detection demo")
    p.add_argument("--edf",    type=Path, required=True, help="Path to EDF file")
    p.add_argument("--delay",  type=float, default=0.0,
                   help="Seconds to sleep between chunks (0 = as fast as possible)")
    p.add_argument("--chunk",  type=int, default=256,
                   help="Chunk size in samples (default=256 = 1s at 256Hz)")
    p.add_argument("--thresh", type=float, default=settings.SEIZURE_THRESHOLD)
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    # ── Load EEG ──────────────────────────────────────────────────────────────
    loader = EDFLoader(target_sfreq=settings.SAMPLING_FREQ)
    record = loader.load(args.edf)

    logger.info("File   : {}", args.edf.name)
    logger.info("Channels: {}", len(record.channel_names))
    logger.info("Duration: {:.1f}s", record.duration_sec)
    logger.info("Seizures annotated: {}", len(record.seizures))
    for s in record.seizures:
        logger.info("  Seizure [{:.1f}s — {:.1f}s]", s.start_sec, s.end_sec)

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        cnn_model = registry.load_cnn()
    except FileNotFoundError:
        logger.error(
            "No trained model found. Run: python scripts/train.py --model cnn"
        )
        sys.exit(1)

    wrapper = ModelWrapper(model=cnn_model)
    service = InferenceService(
        model_wrapper=wrapper,
        threshold=args.thresh,
    )

    # ── Stream ────────────────────────────────────────────────────────────────
    logger.info(
        "\nStarting stream simulation | chunk={}s | threshold={}",
        args.chunk / settings.SAMPLING_FREQ,
        args.thresh,
    )

    all_results = await simulate_realtime_stream(
        service,
        signals=record.signals,
        chunk_size=args.chunk,
        sleep_seconds=args.delay,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    n_alerts = sum(r.is_seizure for r in all_results)
    logger.info("\n── Stream Summary ─────────────────────────────────")
    logger.info("Total windows processed : {}", len(all_results))
    logger.info("Seizure alerts triggered: {}", n_alerts)
    logger.info("Alert rate              : {:.2f}%", 100 * n_alerts / max(len(all_results), 1))

    if n_alerts:
        logger.info("\nAlert timestamps:")
        for r in all_results:
            if r.is_seizure:
                sample_sec = (r.window_start_sample or 0) / settings.SAMPLING_FREQ
                logger.warning(
                    "  🚨 ALERT at {:.1f}s | confidence={:.4f}",
                    sample_sec, r.confidence,
                )

    # Compare against ground truth annotations
    if record.seizures:
        logger.info("\nGround truth seizure windows:")
        for s in record.seizures:
            logger.info("  [{:.1f}s – {:.1f}s]", s.start_sec, s.end_sec)


if __name__ == "__main__":
    asyncio.run(main())
