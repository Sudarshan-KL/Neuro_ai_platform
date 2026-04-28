"""
app/core/logging.py
-------------------
Structured logging via loguru.
Import `logger` everywhere — never use print() in production code.
"""

import sys
from pathlib import Path

from loguru import logger

from app.core.config import settings


def setup_logging() -> None:
    """Configure loguru sinks: stderr + rotating file."""
    logger.remove()  # remove default handler

    # Human-readable console output
    logger.add(
        sys.stderr,
        format=settings.LOG_FORMAT,
        level=settings.LOG_LEVEL,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # Persistent log file with rotation
    log_file = settings.LOG_DIR / "neuro_ai_{time:YYYY-MM-DD}.log"
    logger.add(
        str(log_file),
        format=settings.LOG_FORMAT,
        level=settings.LOG_LEVEL,
        rotation="00:00",      # new file each day
        retention="30 days",
        compression="gz",
        backtrace=True,
        diagnose=False,        # no sensitive data in files
        enqueue=True,          # thread-safe async writing
    )

    logger.info(
        "Logging initialised | level={} | log_dir={}",
        settings.LOG_LEVEL,
        settings.LOG_DIR,
    )


# Initialise on import so the logger is ready when any module imports it
setup_logging()

__all__ = ["logger"]
