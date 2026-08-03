"""
console.py — Terminal/stdio helpers for Sack Counter v23.

Both entry and exit mode print box-drawing characters and arrows, which
raise ``UnicodeEncodeError`` on a Windows console still using a legacy
code page.  :func:`force_utf8_stdio` rewraps the streams to fix that.

Why this is not done at import time
-----------------------------------
``main.py`` and ``exit/exit_main.py`` used to rebind ``sys.stdout`` and
``sys.stderr`` as a module-level side effect.  Merely importing the
package — in a test runner, a notebook, or any host application that
embeds it — silently replaced that process's streams, which is
surprising and can break the importer's own output handling (pytest's
capture, for one).  Rewrapping is now an explicit call made by the CLI
entry points, and it is idempotent.
"""

from __future__ import annotations

import io
import sys

_ALREADY_WRAPPED = "_sack_counter_utf8"


def force_utf8_stdio() -> None:
    """
    Rewrap ``sys.stdout`` / ``sys.stderr`` as UTF-8 with replacement.

    Safe to call more than once — the second call is a no-op.  Streams
    without a binary ``buffer`` (already-wrapped test doubles, pytest's
    captured streams) are left untouched.
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None or getattr(stream, _ALREADY_WRAPPED, False):
            continue
        buffer = getattr(stream, "buffer", None)
        if buffer is None:
            continue
        if (getattr(stream, "encoding", "") or "").lower().replace("-", "") == "utf8":
            setattr(stream, _ALREADY_WRAPPED, True)
            continue
        wrapped = io.TextIOWrapper(buffer, encoding="utf-8", errors="replace")
        try:
            wrapped._sack_counter_utf8 = True   # type: ignore[attr-defined]
        except AttributeError:
            pass
        setattr(sys, name, wrapped)
