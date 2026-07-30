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

from ..confidence import confidence_class
from .state_machine import SackState

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

    # ── Build in-window predicate ─────────────────────────────
    # v22: projection-based window along door normal.
    # Corridor side  → negative projection
    # Room side      → positive projection
    # freeze_edge    → small positive projection just inside the door

    peak_window_px = cfg["peak_window_px"]
    peak_freeze_px = cfg["peak_freeze_px"]

    # approach_open is on the CORRIDOR side (negative proj)
    approach_open_proj = -float(peak_window_px)
    # freeze is just slightly past the centroid toward the room side
    freeze_proj = float(peak_freeze_px)

    def _in_window_proj(cx: int, cy: int) -> bool:
        """True if projection is between approach_open and freeze edge."""
        proj = door.project_onto_normal(cx, cy)
        return approach_open_proj <= proj <= freeze_proj

    # Fallback predicate (v21 x-coord logic) kept for legacy/headless use
    def _in_window_x(pid_cx: int) -> bool:
        door_cx       = door.centroid[0]
        approach_side = cfg.get("door_approach_side", "right")
        if approach_side == "right":
            approach_open = door_cx + peak_window_px
            freeze_edge   = door_cx - peak_freeze_px
            return freeze_edge < pid_cx <= approach_open
        else:
            approach_open = door_cx - peak_window_px
            freeze_edge   = door_cx + peak_freeze_px
            return approach_open <= pid_cx < freeze_edge

    use_normal = hasattr(door, "normal_vec") and door.normal_vec is not None

    def _in_window(pid: int) -> bool:
        cx = state["person_prev_cx"].get(pid, 0)
        if use_normal:
            cy = _get_person_cy(pid, state)
            return _in_window_proj(cx, cy)
        return _in_window_x(cx)

    stamp_thresh = max(cfg["peak_min_assoc_score"] + 0.1, 0.6)

    for pid in state["confirmed_persons"]:
        if pid in state["persons_past_door"]:
            continue
        if not _in_window(pid):
            continue
        state["person_approach_fn"].setdefault(pid, fn)

        # BUG1-B: stamp sacks to this carrier on first high-confidence hit
        for (sid, x1, y1, x2, y2, cx, cy, sc) in confirmed_sack_list:
            if (owner_map.get(sid) == pid
                    and scores_map.get(sid, 0.0) >= stamp_thresh
                    and sc >= cfg["peak_conf_sack_floor"]
                    and scores_map.get(sid, 0.0) >= cfg["peak_min_assoc_score"]
                    and sid not in state["sack_carrier_stamp"]
                    and sid not in state["just_evicted_sacks"]
                    and sid not in state["delivered_sacks"]
                    ):
                old_stamp = state["sack_carrier_stamp"].get(sid)
                if old_stamp is not None and old_stamp != pid:
                    logger.debug(
                        "STAMP TRANSFER sack#%d %d -> P#%d  frame=%d",
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
            cx = state["person_prev_cx"].get(pid, 0)
            logger.debug(
                "PEAK UPDATE P#%d  peak=%d  cx=%d  frame=%d",
                pid, clean_sacks, cx, fn,
            )


def _get_person_cy(pid: int, state) -> int:
    """Return cy from person_boxes, or 0."""
    box = state["person_boxes"].get(pid)
    if box is None:
        return 0
    x1, y1, x2, y2 = box
    return (y1 + y2) // 2
