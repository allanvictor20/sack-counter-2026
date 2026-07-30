"""
person_tracker.py — Person tracking helpers for Sack Counter v22.

v22 adds:
  - cleanup_door_histories() call on timeout to prevent memory leaks in
    person_position_history, door_candidates, and person_peak_count.

Functions
---------
update_persons  : Process new detections, apply ReID, handle re-entry.
timeout_persons : Age out missing persons, snapshot orphaned peaks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .orchestrator import PipelineSession


def update_persons(
    raw_persons: list[tuple],
    session: "PipelineSession",
    fn: int,
) -> set[int]:
    """
    Update person tracking state for one frame.

    For each raw detection:
    - Computes velocity and appearance embedding.
    - Runs ReID re-linking via IdentityRelinker.
    - Confirms persons that have been seen for enough frames.
    - Updates per-person position history (used by door_crossing.py for
      direction confirmation).
    - In v21+, gate-based re-entry and position-exit logic has been removed;
      persons_past_door is managed entirely by door_crossing.py.

    Args:
        raw_persons: Detections as (pid, x1, y1, x2, y2, score).
        session:     Active PipelineSession.
        fn:          Current frame number.

    Returns:
        Set of canonical person IDs active this frame.
    """
    state          = session.state
    cfg            = session.cfg
    relinker       = session.relinker
    ownership_mem  = session.ownership_mem
    embedder       = session.embedder
    confirm_frames = session.confirm_frames
    logger         = session.logger

    active_pids: set[int] = set()

    for (pid, x1, y1, x2, y2, sc) in raw_persons:
        cx   = (x1 + x2) // 2
        vx   = float(cx - state["person_prev_cx"].get(pid, cx))
        area = (x2 - x1) * (y2 - y1)
        emb  = embedder.embed(state["frame"], x1, y1, x2, y2)

        can_pid = relinker.try_relink(
            pid, cx, (y1 + y2) // 2, vx, 0.0, area, emb, fn, logger
        )

        state["person_embeddings"][can_pid] = emb
        state["person_prev_cx"][can_pid]    = cx
        state["person_velocity"][can_pid]   = (vx, 0.0)
        state["person_boxes"][can_pid]      = (x1, y1, x2, y2)
        state["person_hit"][can_pid]       += 1
        state["person_miss"][can_pid]       = 0
        active_pids.add(can_pid)

        if state["person_hit"][can_pid] >= confirm_frames:
            state["confirmed_persons"].add(can_pid)

        # Update position history for direction confirmation in door_crossing.py
        from collections import deque as _deque
        hist = state.persons.person_position_history.setdefault(
            can_pid, _deque(maxlen=cfg.get("direction_lookback_frames", 8) + 2)
        )
        hist.append((cx, (y1 + y2) // 2))

    return active_pids


def timeout_persons(
    active_pids: set[int],
    session: "PipelineSession",
    fn: int,
) -> None:
    """
    Age out confirmed persons not seen this frame and evict their ownerships.

    v22: Also calls cleanup_door_histories(pid) for every timed-out person
    to prevent unbounded growth of person_position_history, door_candidates,
    and person_peak_count on long videos.

    Args:
        active_pids: Person IDs seen this frame.
        session:     Active PipelineSession.
        fn:          Current frame number.
    """
    state         = session.state
    relinker      = session.relinker
    ownership_mem = session.ownership_mem
    miss_frames   = session.miss_frames
    logger        = session.logger

    for pid in list(state["confirmed_persons"]):
        if pid in active_pids:
            continue
        state["person_miss"][pid] += 1
        if state["person_miss"][pid] <= miss_frames:
            continue

        cx       = state["person_prev_cx"].get(pid, 0)
        vx, vy   = state["person_velocity"].get(pid, (0.0, 0.0))
        emb      = state["person_embeddings"].get(pid)

        # BUG #7 FIX: recover cy and area from person_boxes
        box = state["person_boxes"].get(pid)
        if box is not None:
            x1, y1, x2, y2 = box
            cy   = (y1 + y2) // 2
            area = float((x2 - x1) * (y2 - y1))
        else:
            cy   = 0
            area = 0.0

        relinker.register_lost(pid, cx, cy, vx, vy, area, emb, fn)

        # BUG2: snapshot peak before discarding
        stamped_count   = sum(1 for v in state["sack_carrier_stamp"].values()
                              if v == pid)
        peak_at_timeout = state["person_peak_count"].get(pid, 0) or stamped_count
        if peak_at_timeout > 0 and pid not in state["person_peak_delivery"]:
            state["orphaned_peaks"][pid] = {
                "peak":       peak_at_timeout,
                "expires_at": fn + miss_frames,
            }
            owned_at_timeout = [
                sid for sid, _pid in ownership_mem.iter_by_owner()
                if _pid == pid
                and state["sack_carrier_stamp"].get(sid) == pid
            ]
            for sid in owned_at_timeout:
                state["orphaned_sack_owner"][sid] = pid

            logger.info(
                "P#%d timed out — orphaned peak=%d sacks=%s "
                "expires_at=%d  frame=%d",
                pid, peak_at_timeout, owned_at_timeout,
                fn + miss_frames, fn,
            )

            # BUG #4 FIX: also register under canonical ID
            canonical_pid = relinker.canonical(pid)
            if canonical_pid != pid:
                if canonical_pid not in state["orphaned_peaks"]:
                    state["orphaned_peaks"][canonical_pid] = (
                        state["orphaned_peaks"][pid].copy()
                    )
                    logger.debug(
                        "Orphaned peak also registered under canonical "
                        "P#%d  frame=%d", canonical_pid, fn,
                    )
                for sid in owned_at_timeout:
                    if state["orphaned_sack_owner"].get(sid) == pid:
                        state["orphaned_sack_owner"][sid] = canonical_pid
                        logger.debug(
                            "orphaned_sack_owner sack#%d updated to canonical "
                            "P#%d  frame=%d", sid, canonical_pid, fn,
                        )

        state["confirmed_persons"].discard(pid)
        state["person_boxes"].pop(pid, None)

        # FIX 1: evict ghost ownership on timeout
        evicted = [sid for sid, _pid in ownership_mem.iter_by_owner()
                   if _pid == pid]
        for sid in evicted:
            ownership_mem.clear(sid)
            state["sack_carrier_stamp"].pop(sid, None)
            logger.debug(
                "EVICT ownership sack#%d (owner P#%d timed out)  frame=%d",
                sid, pid, fn,
            )
        if evicted:
            logger.info(
                "P#%d timed out — evicted ownership of %d sack(s): %s  frame=%d",
                pid, len(evicted), evicted, fn,
            )

        # v22: clean up door tracking histories to prevent memory leaks
        from .door_crossing import cleanup_door_histories
        cleanup_door_histories(pid, state)
        logger.debug(
            "P#%d door histories cleaned up  frame=%d", pid, fn,
        )
