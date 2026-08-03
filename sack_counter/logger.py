"""
logger.py — Logging setup for Sack Counter v20.

Provides structured logging with configurable levels:
    DEBUG   — Frame-by-frame assignment / stamp / peak details
    INFO    — Delivery counts, gate crossings, re-entry events
    WARNING — Low ownership confidence anomalies
    ERROR   — Detection failures, video read errors
"""

import logging
import os
from pathlib import Path


def setup_logger(
    name:       str  = "sack_counter",
    log_path:   str  = "sack_counter.log",
    file_level: int  = logging.DEBUG,
    console_level: int = logging.WARNING,
) -> logging.Logger:
    """
    Configure and return the named logger.

    Attaches two handlers:
        * FileHandler  — writes all levels >= file_level to *log_path*.
        * StreamHandler — writes levels >= console_level to stderr so
          the terminal stays clean during normal runs.

    Args:
        name:          Logger name (default ``"sack_counter"``).
        log_path:      Path to the log file.
        file_level:    Minimum level written to the log file.
        console_level: Minimum level echoed to the console.

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # capture everything; handlers filter

    if logger.handlers:
        return logger  # already set up (e.g. multiple calls in tests)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # File handler — full detail
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler — warnings+ only (keeps stdout clean)
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger
