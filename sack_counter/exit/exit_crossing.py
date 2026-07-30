"""
exit_crossing.py — Door crossing detection for exiting sacks.

Logic — REDESIGNED
-------------------
The original design tried to detect a ROOM-side -> CORRIDOR-side
projection transition, the same way entry mode detects carriers walking
through the door. That design is structurally broken for exit mode:

    The room interior has bad lighting and sacks are NEVER visible while
    still inside the room (confirmed early in this project). This means
    a sack's FIRST EVER confirmed detection already happens at or past
    the door threshold — there is no earlier "room-side" frame to compare
    against. `prev_proj` and `curr_proj` start identical on first sight,
    so the transition condition (prev > threshold AND curr <= threshold)
    can mathematically never fire. This was the actual cause of the door
    polygon detecting zero crossings, even at correctly lowered confidence
    thresholds.

Corrected trigger
------------------
A sack is registered as a TENTATIVE crossing the first time it is
confirmed AND its position is inside the door polygon OR within
`door_proximity_px` of it. This treats "first ever seen near the door"
as the crossing event, since for an exit-only camera angle, the door
area genuinely IS where a sack first becomes observable — there is no
meaningful "before" state to require.

Tentative -> Confirmed
-----------------------
Unchanged from before: a crossing fires as TENTATIVE first, and becomes
CONFIRMED when the same sack (or any new sack) appears stationary in the
landing zone within CONFIRM_TIMEOUT_FRAMES frames. This still protects
against false positives from people standing near the door, since a
person isn't a "sack" detection and won't trigger this at all — but it
also protects against a stray single-frame sack misdetection that never
actually lands anywhere.

Direct land
-----------
If a new sack appears stationary inside the landing zone WITHOUT a prior
tentative crossing (most common case: detection at the door was missed
entirely, or the sack was first detected already inside the landing
zone), it is counted directly from the landing zone module — see
exit_landing.py. This remains the primary, more reliable counting path;
door-zone tentative crossings are a secondary/earlier signal layered on
top, mainly useful for latency (catching the sack sooner) and as a
sanity cross-check via check_discrepancy().
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .exit_state import ExitPipelineState
    from ..door_polygon import DoorPolygon

logger = logging.getLogger(__name__)

# How many frames a tentative crossing waits for landing confirmation
_DEFAULT_CONFIRM_TIMEOUT = 45   # ~1.5 s at 30 fps

# How close (in pixels) to the door polygon boundary a sack's first
# confirmed position must be to register as a tentative crossing, if
# it's not already inside the polygon itself.
_DEFAULT_DOOR_PROXIMITY_PX = 60


def update_exit_crossings(
    state: "ExitPipelineState",
    door: "DoorPolygon",
    fn: int,
    confirm_timeout: int = _DEFAULT_CONFIRM_TIMEOUT,
    cross_threshold_px: float = 20.0,
    door_proximity_px: float = _DEFAULT_DOOR_PROXIMITY_PX,
) -> list[dict]:
    """
    Check each confirmed sack for a "first seen at/near the door" event.

    Called once per frame AFTER sack positions have been updated.

    Args:
        state:              ExitPipelineState (modified in-place).
        door:               Calibrated DoorPolygon.
        fn:                 Current frame number.
        confirm_timeout:    Frames to wait for landing confirmation.
        cross_threshold_px: Unused by the new trigger directly, but kept
                            for backward-compat call signatures and for
                            potential future use distinguishing "deep
                            inside door" vs "just at the edge".
        door_proximity_px:  How close to the door polygon boundary (in
                            pixels) a sack's first position must be to
                            count, if not already inside the polygon.

    Returns:
        List of new tentative crossing event dicts (may be empty).
    """
    new_tentatives: list[dict] = []

    for sid in list(state["confirmed_sacks"]):
        # Already counted — skip
        if sid in state["crossed_sacks"]:
            continue

        # Already has a pending tentative — skip (avoid duplicate entries)
        if sid in state["tentative_crossings"]:
            continue

        # Only trigger on the sack's FIRST-ever recorded projection —
        # i.e. the first frame we have a position for it at all. Once a
        # sack has a prior recorded projection, it's no longer "newly
        # appeared" and should not re-trigger a tentative crossing here
        # (it may already be moving through the landing zone instead).
        already_seen_before = sid in state["sack_prev_proj"]

        cx, cy = state["sack_positions"].get(sid, (-1, -1))
        if cx < 0:
            continue

        curr_proj = door.project_onto_normal(cx, cy)
        state["sack_curr_proj"][sid] = curr_proj
        state["sack_prev_proj"][sid] = curr_proj

        if already_seen_before:
            continue

        # First-ever sighting of this sack ID. Does it appear at/near
        # the door? If so, this is our crossing signal.
        near_door = door.contains(cx, cy) or (abs(curr_proj) <= door_proximity_px)

        if not near_door:
            # Sack first appeared somewhere else entirely (e.g. directly
            # inside the landing zone, far from the door polygon) — that
            # case is handled by exit_landing.py's direct-land path, not
            # here.
            continue

        # Register tentative crossing
        state["tentative_crossings"][sid] = fn
        event = {
            "frame":        fn,
            "sack_id":      sid,
            "proj":         curr_proj,
            "type":         "tentative",
            "trigger":      "first_seen_at_door",
        }
        new_tentatives.append(event)
        logger.info(
            "EXIT-TENTATIVE sack#%d  first_seen_proj=%.1f  frame=%d",
            sid, curr_proj, fn,
        )

    return new_tentatives


def confirm_tentative_crossings(
    state: "ExitPipelineState",
    fn: int,
    confirm_timeout: int = _DEFAULT_CONFIRM_TIMEOUT,
) -> tuple[list[dict], list[dict]]:
    """
    Resolve pending tentative crossings:
      - Confirm those whose sack (or any sack) landed in the landing zone.
      - Cancel those that have timed out without a landing confirmation.

    The landing zone module sets a flag in state["landed_sacks"] when a
    new sack is confirmed stationary.  We check that set here.

    Args:
        state:           ExitPipelineState (modified in-place).
        fn:              Current frame number.
        confirm_timeout: Max frames to wait for landing confirmation.

    Returns:
        (confirmed_events, cancelled_events) — lists of event dicts.
    """
    confirmed: list[dict] = []
    cancelled: list[dict] = []

    for sid in list(state["tentative_crossings"].keys()):
        tentative_fn = state["tentative_crossings"][sid]
        age = fn - tentative_fn

        # Check if landing zone has seen a new sack since this tentative fired
        # We rely on exit_landing.py having added the sid (or any new sid)
        # to landed_sacks.  If the sack ID changed mid-throw, ANY new landed
        # sack within the timeout window counts as confirmation.
        landing_confirmed = sid in state["landed_sacks"]

        if landing_confirmed:
            # Confirm
            state["tentative_crossings"].pop(sid)
            state["crossed_sacks"].add(sid)
            state["total_sacks_out"] = state["total_sacks_out"] + 1
            event = {
                "frame":   fn,
                "sack_id": sid,
                "type":    "confirmed",
                "trigger": "door_crossing+landing",
                "latency": age,
            }
            state["exit_log"].append(event)
            confirmed.append(event)
            logger.info(
                "EXIT-CONFIRMED sack#%d  latency=%d frames  total_out=%d  frame=%d",
                sid, age, state["total_sacks_out"], fn,
            )

        elif age >= confirm_timeout:
            # Cancel — no landing zone hit within timeout
            state["tentative_crossings"].pop(sid)
            event = {
                "frame":   fn,
                "sack_id": sid,
                "type":    "cancelled",
                "trigger": "timeout",
                "age":     age,
            }
            cancelled.append(event)
            logger.info(
                "EXIT-CANCELLED sack#%d  timed_out=%d frames  frame=%d",
                sid, age, fn,
            )

    return confirmed, cancelled


def expire_old_projections(
    state: "ExitPipelineState",
    active_sack_ids: set,
) -> None:
    """
    Remove projection history for sack IDs that are no longer tracked.

    Call once per frame after update_exit_crossings to keep state lean.

    Args:
        state:           ExitPipelineState (modified in-place).
        active_sack_ids: Set of sack IDs detected this frame.
    """
    stale = set(state["sack_prev_proj"].keys()) - active_sack_ids
    for sid in stale:
        state["sack_prev_proj"].pop(sid, None)
        state["sack_curr_proj"].pop(sid, None)
