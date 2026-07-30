"""
exit_tracker.py — Sack detection and track confirmation for exit mode.

In exit mode we only track SACKS (class index 1).  Person detection is
skipped entirely — it is expensive and not needed for Phase 1 exit counting.

Track confirmation
------------------
A sack track is "confirmed" after it has been seen for at least
*confirm_frames* consecutive frames.  Tracks lost for more than
*miss_frames* are evicted.

This mirrors the hit/miss logic in the entry counter's sack_tracker.py
but is self-contained so the exit module has no dependency on the
entry pipeline internals.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .exit_state import ExitPipelineState

logger = logging.getLogger(__name__)


def update_exit_sacks(
    state: "ExitPipelineState",
    raw_detections: list[tuple],
    fn: int,
    confirm_frames: int = 3,
    miss_frames: int = 8,
) -> list[tuple]:
    """
    Update sack tracks from raw YOLO+ByteTrack detections.

    Args:
        state:          ExitPipelineState (modified in-place).
        raw_detections: List of (sid, x1, y1, x2, y2, cx, cy, conf) tuples
                        for all sacks detected this frame by YOLO+ByteTrack.
        fn:             Current frame number.
        confirm_frames: Consecutive frames before a track is confirmed.
        miss_frames:    Consecutive missed frames before a track is evicted.

    Returns:
        List of (sid, x1, y1, x2, y2, cx, cy, conf) for confirmed sacks only.
    """
    detected_ids: set[int] = set()

    for (sid, x1, y1, x2, y2, cx, cy, conf) in raw_detections:
        detected_ids.add(sid)

        # Update position
        state["sack_positions"][sid] = (cx, cy)
        state["sack_boxes"][sid]     = (x1, y1, x2, y2)
        state["sack_conf"][sid]      = conf

        # Hit counter
        state["sack_hit"][sid] = state["sack_hit"].get(sid, 0) + 1
        state["sack_miss"][sid] = 0  # reset miss streak

    # Miss counter for absent tracks
    for sid in list(state["confirmed_sacks"]):
        if sid not in detected_ids:
            state["sack_miss"][sid] = state["sack_miss"].get(sid, 0) + 1
            if state["sack_miss"][sid] > miss_frames:
                _evict_sack(state, sid)

    # Confirm new tracks
    for sid in detected_ids:
        hits = state["sack_hit"].get(sid, 0)
        if hits >= confirm_frames and sid not in state["confirmed_sacks"]:
            state["confirmed_sacks"].add(sid)
            logger.debug("EXIT SACK CONFIRMED  sid=%d  frame=%d", sid, fn)

    # Build confirmed list with current positions
    confirmed: list[tuple] = []
    for sid in state["confirmed_sacks"]:
        pos = state["sack_positions"].get(sid)
        box = state["sack_boxes"].get(sid)
        conf = state["sack_conf"].get(sid, 0.0)
        if pos and box:
            x1, y1, x2, y2 = box
            cx, cy = pos
            confirmed.append((sid, x1, y1, x2, y2, cx, cy, conf))

    return confirmed


def _evict_sack(state: "ExitPipelineState", sid: int) -> None:
    """Remove all tracking state for a lost sack ID."""
    state["confirmed_sacks"].discard(sid)
    state["sack_positions"].pop(sid, None)
    state["sack_boxes"].pop(sid, None)
    state["sack_conf"].pop(sid, None)
    state["sack_hit"].pop(sid, None)
    state["sack_miss"].pop(sid, None)
    state["sack_prev_proj"].pop(sid, None)
    state["sack_curr_proj"].pop(sid, None)
    # Note: crossed_sacks and landed_sacks are NOT cleared —
    # an evicted sack that already triggered a count should stay latched.
    logger.debug("EXIT SACK EVICTED  sid=%d", sid)


def cleanup_stale_tentatives(
    state: "ExitPipelineState",
    fn: int,
    max_age: int = 90,
) -> None:
    """
    Hard-evict tentative crossings older than *max_age* frames that were
    not cleaned up by confirm_tentative_crossings().

    This is a safety net — confirm_tentative_crossings handles normal
    timeout cancellation.  This handles the edge case where a sack ID
    in tentative_crossings was evicted before it could be cancelled.

    Args:
        state:   ExitPipelineState (modified in-place).
        fn:      Current frame number.
        max_age: Maximum age in frames before forced eviction.
    """
    for sid in list(state["tentative_crossings"].keys()):
        age = fn - state["tentative_crossings"][sid]
        if age > max_age:
            state["tentative_crossings"].pop(sid)
            logger.debug(
                "EXIT TENTATIVE HARD-EVICT  sid=%d  age=%d  frame=%d",
                sid, age, fn,
            )
