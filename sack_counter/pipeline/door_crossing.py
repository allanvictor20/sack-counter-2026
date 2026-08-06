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

from collections import deque
from typing import TYPE_CHECKING

from .geometry import person_centroid

if TYPE_CHECKING:
    from .orchestrator import PipelineSession


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

    # cfg[...] not cfg.get(..., 30): DEFAULT_CFG is the single source of
    # truth for this key, and the inline fallback that used to sit here
    # said 30 while the real default was 5 — the exact drift config.py's
    # docstring warns about, on the very key it names.
    threshold = cfg["door_cross_threshold_px"]

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
        point = person_centroid(state, pid) if pid_detected else None
        if point is not None and point[0] >= 0:
            hist = state.persons.person_position_history.setdefault(
                pid, deque(maxlen=cfg["direction_lookback_frames"] + 2)
            )
            hist.append(point)

        # Find sacks stamped to this carrier
        carrier_sacks = [
            s for s in confirmed_sack_list
            if state["sack_carrier_stamp"].get(s[0]) == pid
        ]
        if not carrier_sacks:
            continue

        # Check if any sack has crossed the door threshold.
        #
        # The projection test is SIGNED.  It previously used abs(proj),
        # which also fired for a sack sitting `threshold` px on the
        # CORRIDOR side: is_on_approach_side() returns False for anything
        # inside the polygon, so a carrier merely standing in a wide
        # doorway satisfied both clauses and got committed before they had
        # actually gone through.  Positive projection = room side, which is
        # the only direction that means "delivered".
        crossed = False
        crossing_sid = None
        crossing_proj = 0.0
        for (sid, x1, y1, x2, y2, scx, scy, sc) in carrier_sacks:
            proj = door.project_onto_normal(scx, scy)
            if proj >= threshold:
                crossed = True
                crossing_sid = sid
                crossing_proj = proj
                logger.debug(
                    "SACK-CROSS sack#%d  proj=%.1f  owner=P#%d  frame=%d",
                    sid, proj, pid, fn,
                )
                break

        if not crossed:
            # How close the load got is the first thing anyone asks when a
            # delivery fails to register, so log it at debug rather than
            # accumulating it in an untyped dict on the session object.
            best = max((door.project_onto_normal(s[5], s[6])
                        for s in carrier_sacks), default=None)
            if best is not None:
                logger.debug(
                    "NO-CROSS P#%d  best proj=%.1f of %.1f needed  frame=%d",
                    pid, best, threshold, fn,
                )
            continue

        # Sack crossed — commit this carrier's peak
        _commit_carrier(
            session, pid,
            peak=state["person_peak_count"].get(pid, 0),
            crossing_sid=crossing_sid,
            crossing_proj=crossing_proj,
            fn=fn,
            trigger="sack_crossing",
        )

    # ── Orphan rescue ─────────────────────────────────────────
    # A carrier whose track dies in the doorway (very common — the person
    # is occluded by their own load as they turn through it) is snapshotted
    # by timeout_persons into orphaned_peaks / orphaned_sack_owner.  Until
    # now nothing consumed those snapshots, so the load was silently never
    # counted despite the peak having been measured correctly.  Their sacks
    # usually remain tracked for a while after the person is gone, so we
    # can still commit on the sack's own crossing.
    _commit_orphaned_carriers(session, confirmed_sack_list, fn, threshold)


def _commit_carrier(
    session: "PipelineSession",
    pid: int,
    peak: int,
    crossing_sid: int,
    crossing_proj: float,
    fn: int,
    trigger: str,
) -> None:
    """
    Commit *pid*'s delivery and retire every sack stamped to them.

    Shared by the live-carrier path and the orphan-rescue path so both
    produce identical bookkeeping.

    Args:
        session:       Active PipelineSession.
        pid:           Carrier person ID.
        peak:          Peak sack count measured for this pass.
        crossing_sid:  Sack whose crossing triggered the commit.
        crossing_proj: Its signed projection (logged only).
        fn:            Current frame number.
        trigger:       Value recorded in the delivery log's ``trigger``.
    """
    state  = session.state
    logger = session.logger

    # Clear all sack stamps for this carrier immediately so these same
    # sacks cannot trigger another commit next frame.  just_evicted_sacks
    # additionally blocks re-stamping within this frame.
    retired: list[int] = []
    for s in list(state["sack_carrier_stamp"].keys()):
        if state["sack_carrier_stamp"].get(s) == pid:
            state["sack_carrier_stamp"].pop(s, None)
            state["just_evicted_sacks"].add(s)
            state["delivered_sacks"].add(s)
            retired.append(s)
            logger.debug(
                "POST-COMMIT stamp clear sack#%d  owner=P#%d  frame=%d",
                s, pid, fn,
            )
    if crossing_sid is not None and crossing_sid not in retired:
        retired.append(crossing_sid)

    state["persons_past_door"].add(pid)
    state["persons_past_gate"].add(pid)
    # Record the commit frame so check_door_reentry does not have to
    # rescan the whole delivery log on every frame.
    state["last_commit_frame"][pid] = fn
    state["orphaned_peaks"].pop(pid, None)

    if peak <= 0:
        logger.info(
            "SACK-CROSS P#%d  sack#%d  proj=%.1f  peak=0 — not counted  frame=%d",
            pid, crossing_sid, crossing_proj, fn,
        )
        return

    # Mark the retired sacks as delivered in the lifecycle registry and in
    # the confidence tracker.  Neither was previously being told about a
    # delivery: no sack ever reached SackState.DELIVERED (so the registry's
    # DELIVERED-pruning path was dead and records accumulated forever), and
    # the crossing term of delivery_confidence was permanently 0.
    for s in retired:
        session.sack_sm.mark_delivered(s, fn)
        session.conf_tracker.record_crossing(s)
        state["person_sack_delivered"][pid].add(s)

    state["person_peak_delivery"][pid] = peak
    state["total_sacks_counted"] += peak
    state["delivery_log"].append({
        "frame":                fn,
        "sack_id":              crossing_sid,
        "person_id":            pid,
        "peak_count":           peak,
        "confidence":           1.0,
        "ownership_confidence": 1.0,
        "confidence_class":     "HIGH",
        "high_conf":            True,
        "type":                 "sack",
        "trigger":              trigger,
    })
    logger.info(
        "SACK-CROSS-COMMIT P#%d  peak=%d  sack#%d  proj=%.1f  frame=%d",
        pid, peak, crossing_sid, crossing_proj, fn,
    )
    session.record_delivery(pid, peak, 1.0, fn)

    # Credit any boxes this carrier was holding at the moment of crossing.
    session.box_pipeline.commit_delivery(pid, fn)


def _commit_orphaned_carriers(
    session: "PipelineSession",
    confirmed_sack_list: list[tuple],
    fn: int,
    threshold: float,
) -> None:
    """
    Commit peaks for carriers who timed out before their sack crossed.

    ``timeout_persons`` records ``orphaned_sack_owner[sid] = pid`` and
    ``orphaned_peaks[pid] = {"peak": n, "expires_at": f}`` when a carrier's
    track dies while still holding stamped sacks.  This walks the sacks
    that survived their owner and commits the snapshot when one of them
    crosses the door.

    Args:
        session:             Active PipelineSession.
        confirmed_sack_list: Confirmed, non-ground sacks this frame.
        fn:                  Current frame number.
        threshold:           Signed crossing threshold in pixels.
    """
    state  = session.state
    door   = session.door
    logger = session.logger

    orphan_owner = state["orphaned_sack_owner"]
    if not orphan_owner:
        return

    for (sid, x1, y1, x2, y2, scx, scy, sc) in confirmed_sack_list:
        pid = orphan_owner.get(sid)
        if pid is None or pid in state["persons_past_door"]:
            continue
        if sid in state["delivered_sacks"]:
            continue
        rec = state["orphaned_peaks"].get(pid)
        if rec is None:
            continue
        if door.project_onto_normal(scx, scy) < threshold:
            continue

        logger.info(
            "ORPHAN-RESCUE P#%d  peak=%d  via sack#%d  frame=%d",
            pid, rec["peak"], sid, fn,
        )
        # Re-stamp the orphan's surviving sacks so _commit_carrier retires
        # them all, not just the one that crossed.
        for other_sid, other_pid in list(orphan_owner.items()):
            if other_pid == pid:
                state["sack_carrier_stamp"].setdefault(other_sid, pid)
                orphan_owner.pop(other_sid, None)

        _commit_carrier(
            session, pid,
            peak=rec["peak"],
            crossing_sid=sid,
            crossing_proj=door.project_onto_normal(scx, scy),
            fn=fn,
            trigger="orphan_rescue",
        )


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

    reentry_margin = cfg["reentry_margin_px"]
    grace_frames   = cfg["reentry_grace_frames"]

    for pid in list(state["persons_past_door"]):
        # FIX 2: skip if this person was committed within the grace period.
        # Uses the last_commit_frame index rather than rescanning the whole
        # (unbounded, session-long) delivery_log on every single frame.
        last_commit = state["last_commit_frame"].get(pid)
        if last_commit is not None and fn - last_commit <= grace_frames:
            continue

        point = person_centroid(state, pid)
        if point is None or point[0] < 0:
            continue
        cx, cy = point

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
