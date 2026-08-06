"""
frame.py — One frame through the entry-mode pipeline.

These stages used to sit inline in ``main.run``, numbered 1–11 in
comments inside a 350-line function.  The numbering was already telling
us they were separate steps; they are now separate functions, in the
same order, doing the same work.

Splitting them is not cosmetic.  The loop could only be exercised with a
real model and a real video, so the stages had no unit tests and bugs in
them were found by watching counts drift.  Each function here takes the
session and a frame number and can be driven on its own.

``advance_frame`` runs the whole sequence and is what the loop calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..assignment import assign_sacks_hungarian
from .counting import update_peak_counts
from .door_crossing import check_door_reentry, update_door_zone
from .geometry import has_approached, person_projection
from .person_tracker import timeout_persons, update_persons
from .sack_tracker import update_sacks
from .state_machine import SackState

if TYPE_CHECKING:
    from .orchestrator import PipelineSession


def track_persons(session: "PipelineSession", raw_persons: list[tuple],
                  fn: int) -> set:
    """
    Stage 2 — update person tracks and feed re-links to analytics.

    Returns:
        The set of person ids detected this frame.
    """
    reid_before = len(session.relinker.reid_events)
    active_pids = update_persons(raw_persons, session, fn)
    timeout_persons(active_pids, session, fn)
    session.relinker.expire_lost(fn)
    # record_reid_event() existed but was never called, so the analytics
    # block reported 0 re-IDs in the same JSON file whose top-level
    # reid_events list was populated.
    for _ in range(len(session.relinker.reid_events) - reid_before):
        session.analytics.record_reid_event()
    return active_pids


def confirmed_sacks_this_frame(session: "PipelineSession",
                               raw_sacks: list[tuple]) -> list[tuple]:
    """Stage 4a — confirmed, non-ground sacks, the only ones assignable."""
    state = session.state
    return [
        s for s in raw_sacks
        if s[0] in state["confirmed_sacks"] and s[0] not in state["ground_sacks"]
    ]


def release_stale_ownership(session: "PipelineSession",
                            confirmed_sack_list: list[tuple], fn: int) -> None:
    """
    Stage 4b — drop ownership held by a carrier who is already through.

    Otherwise a sack left behind stays stamped to someone who has gone,
    and can never be picked up and counted by whoever collects it next.
    """
    state  = session.state
    logger = session.logger
    for (sid, *_rest) in confirmed_sack_list:
        owner = session.ownership_mem.get(sid)
        if owner is not None and owner in state["persons_past_door"]:
            session.ownership_mem.clear(sid)
            state["sack_carrier_stamp"].pop(sid, None)
            logger.debug(
                "STALE-OWNER release: sack#%d owner=P#%d cleared  frame=%d",
                sid, owner, fn,
            )


def assign_sacks(session: "PipelineSession",
                 confirmed_sack_list: list[tuple], fn: int) -> dict:
    """
    Stage 4c — assign sacks to carriers and record ownership confidence.

    Returns:
        ``{"owner": …, "scores": …, "stats": …}`` where owner/scores are
        pruned to sacks actually present this frame.
    """
    state = session.state
    cfg   = session.cfg

    active_persons = {
        pid: box for pid, box in state["person_boxes"].items()
        if pid in state["confirmed_persons"]
    }

    sack_owner, sack_scores, person_load, stats = assign_sacks_hungarian(
        confirmed_sack_list, active_persons,
        session.motion_tracker, session.ownership_mem, cfg,
        sack_embeddings=state["sack_embeddings"],
        person_emb_fn=lambda pid: state["person_embeddings"].get(pid),
        persons_past_gate=state["persons_past_door"],
    )
    state["sack_owner"]  = sack_owner
    state["sack_scores"] = sack_scores
    state["person_load"] = person_load

    _record_ownership_confidence(session, sack_owner, sack_scores, fn)

    # Prune to live sacks, dropping stale ghost-carried entries.
    active = {s[0] for s in confirmed_sack_list}
    return {
        "owner":  {s: p for s, p in sack_owner.items()  if s in active},
        "scores": {s: v for s, v in sack_scores.items() if s in active},
        "stats":  stats,
        "persons": active_persons,
    }


def _record_ownership_confidence(session: "PipelineSession", sack_owner: dict,
                                 sack_scores: dict, fn: int) -> None:
    """Log an anomaly for every assignment made with low confidence."""
    state  = session.state
    cfg    = session.cfg
    logger = session.logger
    threshold = cfg["ownership_anomaly_threshold"]

    for sid, score in sack_scores.items():
        session.conf_tracker.record_ownership(sid, score)
        if score >= threshold:
            continue
        state["anomaly_log"].append({
            "frame": fn, "type": "low_owner_confidence",
            "sack_id": sid, "person_id": sack_owner.get(sid),
            "ownership_confidence": round(score, 3),
        })
        state["n_anomalies"] = len(state["anomaly_log"])
        logger.warning(
            "LOW OWNERSHIP CONF sack#%d -> P#%s score=%.3f  frame=%d",
            sid, sack_owner.get(sid), score, fn,
        )
        session.analytics.record_anomaly()


def update_boxes(session: "PipelineSession", raw_boxes: list[tuple],
                 active_persons: dict, fn: int, src_fps: float) -> dict:
    """
    Stage 5 — run the box pipeline and mark unowned boxes as scenery.

    Returns:
        ``{box_id: person_id}`` for boxes currently held.
    """
    state = session.state
    box_owner = session.box_pipeline.update(
        raw_boxes, active_persons, fn, src_fps, cfg=session.cfg,
    )

    # Visual only: an unowned box is part of the room, not a delivery.
    active_bids = {b[0] for b in raw_boxes}
    for (bid, *_rest) in raw_boxes:
        if box_owner.get(bid) is None:
            state["ground_boxes"].add(bid)
        else:
            state["ground_boxes"].discard(bid)
    state["ground_boxes"] &= active_bids
    return box_owner


def reclaim_peak_state(session: "PipelineSession") -> None:
    """
    Stage 8 — forget the peak of a carrier who is well past the door.

    Guarded on ``persons_past_door`` and measured along the door normal.
    It previously did neither: an x-only threshold with no past-door
    guard could wipe the peak of a carrier still mid-approach.
    """
    state = session.state
    margin = session.cfg["peak_freeze_px"] * 2
    for pid in list(state["person_peak_count"].keys()):
        if pid not in state["persons_past_door"]:
            continue
        proj = person_projection(session.door, state, pid)
        if proj is not None and proj > margin:
            state["person_peak_count"].pop(pid, None)
            state["person_approach_fn"].pop(pid, None)


def expire_orphaned_peaks(session: "PipelineSession", fn: int) -> None:
    """Stage 9 — drop snapshots of carriers whose sacks never crossed."""
    state  = session.state
    logger = session.logger

    expired = [pid for pid, rec in state["orphaned_peaks"].items()
               if fn > rec["expires_at"]]
    if not expired:
        return
    for pid in expired:
        state["orphaned_peaks"].pop(pid)
        logger.debug("Orphaned peak P#%d expired without commit  frame=%d",
                     pid, fn)
    live = set(state["orphaned_peaks"].keys())
    state["orphaned_sack_owner"] = {
        s: p for s, p in state["orphaned_sack_owner"].items() if p in live
    }


def advance_sack_lifecycle(session: "PipelineSession", sack_owner: dict,
                           fn: int) -> None:
    """
    Stage 10 — move carried sacks along their lifecycle.

    Approach is measured by projection onto the door normal, like every
    other geometry test here.  This stage kept an x-versus-centroid
    comparison long after the rest had moved on.
    """
    window = session.cfg["peak_window_px"]
    for sid, owner_pid in sack_owner.items():
        rec = session.sack_sm.get(sid)
        if rec is None:
            continue
        if rec.state in (SackState.CONFIRMED, SackState.PICKED_UP):
            rec.transition(SackState.CARRIED, fn)
        if rec.state == SackState.CARRIED:
            proj = person_projection(session.door, session.state, owner_pid)
            if has_approached(proj, window):
                rec.transition(SackState.APPROACHING, fn)


def advance_frame(session: "PipelineSession", detections, fn: int,
                  src_fps: float) -> dict[str, Any]:
    """
    Run every per-frame stage, in order, for one frame.

    Args:
        session:    Active PipelineSession, already holding this frame.
        detections: This frame's :class:`~sack_counter.detections.FrameDetections`.
        fn:         Frame number.
        src_fps:    Source frame rate.

    Returns:
        ``{"box_owner": …, "stats": …}`` — what drawing and the HUD need.
    """
    state = session.state
    cfg   = session.cfg

    state["raw_sacks"] = detections.sacks
    state["raw_boxes"] = detections.boxes

    active_pids = track_persons(session, detections.persons, fn)
    update_sacks(detections.sacks, session, fn)

    confirmed = confirmed_sacks_this_frame(session, detections.sacks)
    release_stale_ownership(session, confirmed, fn)
    assigned = assign_sacks(session, confirmed, fn)

    box_owner = update_boxes(session, detections.boxes,
                             assigned["persons"], fn, src_fps)

    if cfg["peak_count_enabled"]:
        update_peak_counts(confirmed, session, fn,
                           assigned["owner"], assigned["scores"])

    check_door_reentry(session, fn)

    crossings_before = len(state["delivery_log"])
    update_door_zone(session, fn, active_pids)
    n_crossings = len(state["delivery_log"]) - crossings_before

    if cfg["peak_count_enabled"]:
        reclaim_peak_state(session)
    expire_orphaned_peaks(session, fn)
    advance_sack_lifecycle(session, assigned["owner"], fn)

    return {
        "box_owner": box_owner,
        "stats": {
            "raw":       len(detections.sacks),
            "boxes":     len(detections.boxes),
            "still":     assigned["stats"].get("still", 0),
            "assigned":  assigned["stats"].get("assigned", 0),
            "crossings": n_crossings,
            "ghosts":    len(list(session.ghost_sacks.iter_ghosts())),
        },
    }
