"""
door_crossing.py — Sack-crossing delivery trigger for v22.

Trigger: a sack stamped to a carrier crosses the door threshold
(abs projection onto door normal >= door_cross_threshold_px AND
not on approach side) -> commit that carrier's peak immediately.

Fixes:
- After commit, all sacks stamped to that carrier are cleared so they
  cannot trigger another commit on the next frame.
- check_door_reentry has a grace period of 5 frames after any commit
  so the reentry reset cannot fire immediately after a delivery.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

from ..confidence import confidence_class
from .state_machine import SackState

if TYPE_CHECKING:
    from .orchestrator import PipelineSession
    from ..door_polygon import DoorPolygon


# ── State enum (kept for compatibility with drawing.py) ────────

class DoorState(Enum):
    CORRIDOR     = auto()
    APPROACHING  = auto()
    IN_DOOR      = auto()
    DISAPPEARING = auto()
    COUNTED      = auto()
    ABORTED      = auto()


# ── Per-carrier door state (kept for compatibility) ────────────

@dataclass
class DoorCandidateState:
    pid:                  int
    entered_door_fn:      int
    peak_at_entry:        int
    direction_ok:         bool
    door_state:           DoorState = DoorState.IN_DOOR
    max_miss_streak:      int  = 0
    disappear_started_fn: int  = 0
    last_seen_cx:         int  = 0
    last_seen_cy:         int  = 0
    aborted:              bool = False


# ── Main update functions ──────────────────────────────────────

def update_door_zone(
    session: "PipelineSession",
    fn: int,
    active_pids: set | None = None,
) -> None:
    """
    Commit delivery when a SACK stamped to a carrier crosses the door.

    For each confirmed person not yet past the door:
      1. Find all confirmed sacks stamped to them.
      2. Check if any sack centroid has crossed the door threshold.
      3. If yes, commit that carrier's peak count immediately.
      4. Clear all sack stamps for that carrier so the same sacks
         cannot trigger another commit on subsequent frames.

    Threshold: abs(projection onto door normal) >= door_cross_threshold_px
    AND the sack is not on the approach/corridor side.
    """
    state  = session.state
    door   = session.door
    cfg    = session.cfg
    logger = session.logger

    threshold = cfg.get("door_cross_threshold_px", 30)

    # Get all confirmed non-ground sacks this frame
    confirmed_sack_list = [
        s for s in state.get("raw_sacks", [])
        if s[0] in state["confirmed_sacks"]
        and s[0] not in state["ground_sacks"]
    ]

    for pid in list(state["confirmed_persons"]):
        if pid in state["persons_past_door"]:
            continue

        # Update position history for this person
        pid_detected = (active_pids is None) or (pid in active_pids)
        cx = state["person_prev_cx"].get(pid, -1) if pid_detected else -1
        cy = _get_person_cy(pid, state) if pid_detected else 0
        if cx >= 0:
            hist = state.persons.person_position_history.setdefault(
                pid, deque(maxlen=cfg.get("direction_lookback_frames", 8) + 2)
            )
            hist.append((cx, cy))

        # Find sacks stamped to this carrier
        carrier_sacks = [
            s for s in confirmed_sack_list
            if state["sack_carrier_stamp"].get(s[0]) == pid
        ]
        if not carrier_sacks:
            continue

        # Check if any sack has crossed the door threshold
        crossed = False
        crossing_sid = None
        crossing_proj = 0.0
        for (sid, x1, y1, x2, y2, scx, scy, sc) in carrier_sacks:
            proj = door.project_onto_normal(scx, scy)
            if abs(proj) >= threshold and not door.is_on_approach_side(scx, scy, 0):
                crossed = True
                crossing_sid = sid
                crossing_proj = proj
                logger.debug(
                    "SACK-CROSS sack#%d  proj=%.1f  owner=P#%d  frame=%d",
                    sid, proj, pid, fn,
                )
                break

        if not crossed:
            continue

        # Sack crossed — commit this carrier's peak
        peak = state["person_peak_count"].get(pid, 0)

        # FIX 1: clear all sack stamps for this carrier immediately so
        # these same sacks cannot trigger another commit next frame
        # FIX 1: clear all sack stamps for this carrier immediately
        # Also add to just_evicted_sacks to block re-stamping this frame
        for s in list(state["sack_carrier_stamp"].keys()):
            if state["sack_carrier_stamp"].get(s) == pid:
                state["sack_carrier_stamp"].pop(s, None)
                state["just_evicted_sacks"].add(s)
                state["delivered_sacks"].add(s)
                logger.debug(
                    "POST-COMMIT stamp clear sack#%d  owner=P#%d  frame=%d",
                    s, pid, fn,
        )

        if peak <= 0:
            logger.info(
                "SACK-CROSS P#%d  sack#%d  proj=%.1f  peak=0 — not counted  frame=%d",
                pid, crossing_sid, crossing_proj, fn,
            )
            state["persons_past_door"].add(pid)
            state["persons_past_gate"].add(pid)
            continue

        state["person_peak_delivery"][pid] = peak
        state["persons_past_door"].add(pid)
        state["persons_past_gate"].add(pid)
        state["total_sacks_counted"] += peak
        state["delivery_log"].append({
            "frame":                fn,
            "sack_id":              crossing_sid,
            "person_id":            pid,
            "peak_count":           peak,
            "confidence":           1.0,
            "ownership_confidence": 1.0,
            "confidence_class":     "high",
            "high_conf":            True,
            "type":                 "sack",
            "trigger":              "sack_crossing",
        })
        logger.info(
            "SACK-CROSS-COMMIT P#%d  peak=%d  sack#%d  proj=%.1f  frame=%d",
            pid, peak, crossing_sid, crossing_proj, fn,
        )
        session.record_delivery(pid, peak, 1.0, fn)


def process_door_disappearance(
    session: "PipelineSession",
    fn: int,
) -> int:
    """Kept for compatibility — does nothing in this version."""
    return 0


def check_door_reentry(
    session: "PipelineSession",
    fn: int,
) -> None:
    """
    Reset persons_past_door when a carrier reappears clearly in the corridor.

    FIX 2: grace period of 5 frames after any commit — do not reset a
    person who was just committed this frame or the previous 5 frames.
    This stops the reentry reset from firing immediately after a delivery
    before the sacks have cleared the doorway.
    """
    state  = session.state
    door   = session.door
    cfg    = session.cfg
    logger = session.logger

    reentry_margin = cfg.get("reentry_margin_px", 150)
    grace_frames   = cfg.get("reentry_grace_frames", 5)

    # Build set of pids committed recently (within grace period)
    recent_commits = {
        d["person_id"] for d in state["delivery_log"]
        if fn - d["frame"] <= grace_frames
    }

    for pid in list(state["persons_past_door"]):
        # FIX 2: skip if this person was committed within the grace period
        if pid in recent_commits:
            continue

        cx = state["person_prev_cx"].get(pid, -1)
        if cx < 0:
            continue
        cy = _get_person_cy(pid, state)

        if door.is_on_approach_side(cx, cy, reentry_margin):
            state["persons_past_door"].discard(pid)

            stale = [s for s, p in state["sack_carrier_stamp"].items()
                     if p == pid]
            for s in stale:
                state["sack_carrier_stamp"].pop(s, None)
            state["person_peak_delivery"].pop(pid, None)
            state["person_peak_count"][pid] = 0
            
            logger.info(
                "DOOR-REENTRY P#%d back in corridor — reset  frame=%d",
                pid, fn,
            )


def cleanup_door_histories(pid: int, state) -> None:
    """
    Remove per-pid door tracking state when a person times out.
    person_peak_count is intentionally NOT removed — ReID can restore
    the same pid and we don't want to lose the peak.
    """
    state.persons.person_position_history.pop(pid, None)
    state["door_candidates"].pop(pid, None)


# ── Private helpers ────────────────────────────────────────────

def _get_person_cy(pid: int, state) -> int:
    """Return the vertical centroid for pid from person_boxes, or 0."""
    box = state["person_boxes"].get(pid)
    if box is None:
        return 0
    x1, y1, x2, y2 = box
    return (y1 + y2) // 2


def _commit_door_delivery(
    pid: int,
    peak: int,
    session: "PipelineSession",
    fn: int,
) -> int:
    """Kept for compatibility — not called in this version."""
    return 0


def _find_carrier_sack(pid: int, state) -> Optional[int]:
    """Return the sack ID most recently stamped to pid, or None."""
    stamped = [s for s, p in state["sack_carrier_stamp"].items() if p == pid]
    if not stamped:
        return None
    return max(stamped, key=lambda s: state["sack_scores"].get(s, 0.0), default=None)