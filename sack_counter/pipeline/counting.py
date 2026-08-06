"""
counting.py — Peak counting logic for Sack Counter v22.

v22: Peak window boundaries are now projected onto the door normal vector
instead of using raw x coordinates.  This makes the counting zone
camera-angle independent — a diagonal door no longer requires manual
tuning of peak_window_px in screen-space.

Functions
---------
update_peak_counts : Track per-person sack peak inside approach window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .geometry import in_approach_window, person_projection

if TYPE_CHECKING:
    from .orchestrator import PipelineSession


def update_peak_counts(
    confirmed_sack_list: list[tuple],
    session: "PipelineSession",
    fn: int,
    sack_owner_override: dict | None = None,
    sack_scores_override: dict | None = None,
) -> None:
    """
    Update per-person peak sack count inside the door approach window.

    v22 change: window boundaries are expressed in portal coordinates —
    projection of the carrier centroid onto the door normal vector.

      approach_open_proj  : how far on the CORRIDOR side to start tracking
                            (negative projection = corridor side)
      freeze_proj         : how close to the door to freeze the peak

    A carrier is in the window when:
        approach_open_proj <= projection(cx, cy) <= -freeze_proj

    Both values are signed projections along normal_vec, so the geometry
    is independent of camera angle.

    Fallback: if normal_vec is unavailable (legacy or headless mode without
    calibration), falls back to the v21 x-coordinate window logic.

    Args:
        confirmed_sack_list:  Ground-filtered confirmed sack list.
        session:              Active PipelineSession.
        fn:                   Current frame number.
        sack_owner_override:  Clean sack→person dict.
        sack_scores_override: Corresponding association scores.
    """
    state  = session.state
    cfg    = session.cfg
    logger = session.logger
    door   = session.door

    owner_map  = sack_owner_override  or state["sack_owner"]
    scores_map = sack_scores_override or state["sack_scores"]

    # ── In-window predicate ───────────────────────────────────
    # Projection along the door normal: corridor side is negative, room
    # side positive, and the window closes just past the plane so a
    # carrier's peak freezes while they are in the doorway.
    peak_window_px = cfg["peak_window_px"]
    peak_freeze_px = cfg["peak_freeze_px"]

    def _in_window(pid: int) -> bool:
        return in_approach_window(
            person_projection(door, state, pid),
            peak_window_px, peak_freeze_px,
        )

    stamp_thresh = cfg["peak_stamp_score"]

    for pid in state["confirmed_persons"]:
        if pid in state["persons_past_door"]:
            continue
        if not _in_window(pid):
            continue
        state["person_approach_fn"].setdefault(pid, fn)

        # BUG1-B: stamp sacks to this carrier on first high-confidence hit.
        #
        # The eligibility test deliberately does NOT require the sack to be
        # unstamped.  It used to (`sid not in sack_carrier_stamp`), which
        # made the STAMP TRANSFER branch below unreachable: a sack could
        # never move to a second carrier, so a handover between two workers
        # stayed credited to whoever picked it up first.  Transfer is safe
        # to allow here because owner_map is already the output of
        # OwnershipMemory's hysteresis — a new pid only wins after beating
        # the incumbent by ownership_switch_margin.
        for (sid, x1, y1, x2, y2, cx, cy, sc) in confirmed_sack_list:
            if (owner_map.get(sid) == pid
                    and scores_map.get(sid, 0.0) >= stamp_thresh
                    and sc >= cfg["peak_conf_sack_floor"]
                    and scores_map.get(sid, 0.0) >= cfg["peak_min_assoc_score"]
                    and state["sack_carrier_stamp"].get(sid) != pid
                    and sid not in state["just_evicted_sacks"]
                    and sid not in state["delivered_sacks"]
                    ):
                old_stamp = state["sack_carrier_stamp"].get(sid)
                if old_stamp is not None:
                    logger.debug(
                        "STAMP TRANSFER sack#%d P#%d -> P#%d  frame=%d",
                        sid, old_stamp, pid, fn,
                    )
                    state["person_peak_count"][old_stamp] = 0
                    state["person_approach_fn"].pop(old_stamp, None)
                state["sack_carrier_stamp"][sid] = pid
                logger.debug(
                    "STAMP sack#%d -> P#%d score=%.3f  frame=%d",
                    sid, pid, scores_map.get(sid, 0.0), fn,
                )

        # Count sacks cleanly stamped to this carrier
        clean_sacks = sum(
            1 for (sid, x1, y1, x2, y2, cx, cy, sc) in confirmed_sack_list
            if (owner_map.get(sid) == pid
                and scores_map.get(sid, 0.0) >= cfg["peak_min_assoc_score"]
                and sc >= cfg["peak_conf_sack_floor"]
                and sid not in state["just_evicted_sacks"]
                and sid not in state["delivered_sacks"]
                and state["sack_carrier_stamp"].get(sid) == pid)
        )
        if clean_sacks > state["person_peak_count"][pid]:
            state["person_peak_count"][pid] = clean_sacks
            logger.debug(
                "PEAK UPDATE P#%d  peak=%d  proj=%s  frame=%d",
                pid, clean_sacks,
                person_projection(door, state, pid), fn,
            )
