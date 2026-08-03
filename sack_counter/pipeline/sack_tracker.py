"""
sack_tracker.py — Sack tracking helpers for Sack Counter v20.

Handles per-frame sack state updates, ground-sack suppression,
GroundMemory integration, and state machine transitions.

Functions
---------
update_sacks : Process sack detections, drive ground/motion logic, expire lost tracks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .state_machine import SackState

if TYPE_CHECKING:
    from .orchestrator import PipelineSession


def update_sacks(
    raw_sacks: list[tuple],
    session: PipelineSession,
    fn: int,
) -> set[int]:
    """
    Update sack tracking state and ground-sack suppression for one frame.

    For each raw sack detection:
    - Updates motion tracker and ghost-sack position history.
    - Advances the SackStateMachine (DETECTED → CONFIRMED → ON_GROUND /
      PICKED_UP).
    - Extracts an appearance embedding for use in assignment.
    - Drives GroundMemory: marks still sacks as grounded, confirms pickup
      only after ``lift_owner_stable_frames`` consecutive moving frames.

    For confirmed sacks absent this frame:
    - Increments miss counter; expires tracking state once miss_frames exceeded.
    - Cleans up ownership, confidence, ground, stamp, and orphan entries.

    Args:
        raw_sacks: Detections as (sid, x1, y1, x2, y2, cx, cy, score).
        session:   Active PipelineSession.
        fn:        Current frame number.

    Returns:
        Set of sack IDs active (detected) this frame.
    """
    state           = session.state
    embedder        = session.embedder
    motion_tracker  = session.motion_tracker
    ghost_sacks     = session.ghost_sacks
    ownership_mem   = session.ownership_mem
    conf_tracker    = session.conf_tracker
    ground_memory   = session.ground_memory
    sack_sm         = session.sack_sm
    confirm_frames  = session.confirm_frames
    miss_frames     = session.miss_frames
    suppress_frames = session.suppress_frames

    active_sids: set[int] = set()

    for (sid, x1, y1, x2, y2, cx, cy, sc) in raw_sacks:
        motion_tracker.update(sid, cx, cy)
        ghost_sacks.record_position(sid, cx, cy)
        state["sack_positions"][sid]  = (cx, cy)
        state["sack_boxes_dict"][sid] = (x1, y1, x2, y2)
        state["sack_hit"][sid]       += 1
        state["sack_miss"][sid]       = 0
        active_sids.add(sid)

        # State machine record
        rec           = sack_sm.get_or_create(sid, fn)
        rec.last_seen = fn
        rec.position  = (cx, cy)

        if state["sack_hit"][sid] >= confirm_frames:
            state["confirmed_sacks"].add(sid)
            if rec.state == SackState.DETECTED:
                rec.transition(SackState.CONFIRMED, fn)

        # Appearance embedding
        emb = embedder.embed(state["frame"], x1, y1, x2, y2)
        state["sack_embeddings"][sid] = emb
        conf_tracker.record_detection(sid, sc)

        # ── Ground-sack suppression + GroundMemory ────────────
        if motion_tracker.is_still(sid):
            state["sack_still_frames"][sid] = (
                state["sack_still_frames"].get(sid, 0) + 1
            )
            if state["sack_still_frames"].get(sid, 0) >= suppress_frames:
                state["ground_sacks"].add(sid)
                ground_memory.mark_grounded(sid, (cx, cy), fn, emb)
                if rec.state not in (SackState.ON_GROUND, SackState.DELIVERED):
                    rec.transition(SackState.ON_GROUND, fn)
            else:
                # Not yet suppressed but still still — keep updating position
                ground_memory.mark_grounded(sid, (cx, cy), fn, emb)
        else:
            state["sack_still_frames"][sid] = 0
            was_ground = sid in state["ground_sacks"]
            state["ground_sacks"].discard(sid)

            if was_ground:
                picked_up = ground_memory.mark_moving(sid, fn)
                if picked_up and rec.state == SackState.ON_GROUND:
                    rec.transition(SackState.PICKED_UP, fn)
            else:
                ground_memory.mark_moving(sid, fn)

    # ── Expire lost tracks ────────────────────────────────────
    # Both confirmed AND never-confirmed IDs are aged out here.  Only
    # confirmed_sacks used to be walked, so a one-frame false positive that
    # never reached confirm_frames kept its entries in sack_hit,
    # sack_positions, sack_boxes_dict and sack_embeddings (a 576-float32
    # vector each) for the rest of the session.
    tracked_sids = set(state["sack_hit"]) | set(state["confirmed_sacks"])
    for sid in tracked_sids:
        if sid in active_sids:
            continue
        state["sack_miss"][sid] += 1
        if state["sack_miss"][sid] > miss_frames:
            _expire_sack(state, sid, ownership_mem, conf_tracker)

    # ── Global cleanup ────────────────────────────────────────
    ghost_sacks.update(
        active_sids, state["sack_owner"],
        state["sack_positions"], state["sack_boxes_dict"],
    )
    motion_tracker.cleanup(active_sids)
    ownership_mem.cleanup(active_sids)
    ground_memory.cleanup(active_sids, fn)
    sack_sm.cleanup(active_sids, fn)
    # Ghost history is keyed on every ID still being tracked (not just the
    # ones visible this frame), so an occluded sack keeps its velocity.
    ghost_sacks.cleanup(set(state["sack_positions"]))

    return active_sids


def _expire_sack(state, sid: int, ownership_mem, conf_tracker) -> None:
    """
    Drop every per-sack entry for a track that has been lost too long.

    BUG #5 FIX: the per-entry methods are called directly rather than
    ``cleanup({sid})`` — ``cleanup`` expects the set of sacks to KEEP, so
    passing the expiring sack was semantically backwards.

    ``delivered_sacks`` is cleared here too.  It is the latch that stops a
    counted sack from being re-counted, but it was never pruned: since
    ByteTrack recycles tracker IDs, a genuinely new sack that inherited a
    retired ID could never be stamped or counted again, so the count drifted
    progressively low over a long session.  Once the ID has been absent for
    miss_frames it is no longer the sack that was delivered, so the latch
    must go with the rest of the track's state.

    Args:
        state:         PipelineState being mutated.
        sid:           Sack tracker ID to expire.
        ownership_mem: OwnershipMemory to clear.
        conf_tracker:  ConfidenceTracker to prune.
    """
    state["confirmed_sacks"].discard(sid)
    ownership_mem.clear(sid)
    conf_tracker.remove(sid)
    state["sack_hit"].pop(sid, None)
    state["sack_miss"].pop(sid, None)
    state["sack_positions"].pop(sid, None)
    state["sack_boxes_dict"].pop(sid, None)
    state["sack_embeddings"].pop(sid, None)
    state["sack_still_frames"].pop(sid, None)
    state["ground_sacks"].discard(sid)
    state["sack_carrier_stamp"].pop(sid, None)
    state["orphaned_sack_owner"].pop(sid, None)
    state["delivered_sacks"].discard(sid)
