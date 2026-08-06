"""
session_log.py — Naming for the JSON log each session writes.

Entry mode used to write ``delivery_log_<version>.json`` — a constant
name, so every run silently overwrote the one before it and the History
screen could only ever list a single entry session no matter how many
had been run.  Exit mode already timestamped its logs.  Both now go
through :func:`session_log_path`, so the two modes agree and a session's
record survives the next session.

The name still starts with the prefix the History screen globs
(``delivery_log_*.json`` / ``exit_log_*.json``), so logs written by
older versions keep showing up.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .version import VERSION_TAG

#: ``strftime`` pattern for the timestamp segment.  Sorts lexicographically
#: in chronological order, which is what the History listing relies on when
#: two logs share a modification time.
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def session_log_path(prefix: str, when: datetime | None = None,
                     directory: Path | str = ".") -> str:
    """
    Build a log file name for one session that does not already exist.

    The timestamp has one-second resolution, so two sessions finishing
    inside the same second would otherwise still collide — exactly the
    overwrite this function exists to prevent, just rarer and harder to
    notice.  A numeric suffix disambiguates those.

    Args:
        prefix:    ``"delivery_log"`` for entry mode, ``"exit_log"`` for exit.
        when:      Session end time.  Defaults to now.
        directory: Where the log will be written, checked for collisions.

    Returns:
        A file name (not a path), e.g.
        ``delivery_log_v23_20260804_142233.json``.
    """
    stamp = (when or datetime.now()).strftime(TIMESTAMP_FORMAT)
    base  = f"{prefix}_{VERSION_TAG}_{stamp}"
    folder = Path(directory)

    candidate = f"{base}.json"
    suffix = 2
    while (folder / candidate).exists():
        candidate = f"{base}_{suffix}.json"
        suffix += 1
    return candidate
