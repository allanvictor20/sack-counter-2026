"""
assignment.py — Sack-to-person assignment helpers for Sack Counter v20.

Functions:
    carry_zone_bounds(person_box, cfg)   → (x1, y1, x2, y2)
    association_score(...)               → float  [0, 1]
    assign_sacks_hungarian(...)          → (sack_owner, sack_scores,
                                            person_load, stats)

v20 changes
-----------
* ``assign_sacks_hungarian`` now accepts a ``persons_past_gate`` set and
  filters those IDs from ``person_boxes`` before building the cost matrix.
  This makes the contract explicit at the function boundary — previously
  the caller was solely responsible for pre-filtering, so a missed filter
  could silently let a bystander past the gate win Hungarian assignment
  (the P#457 bug class).
"""

import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

from .embedder import DeepEmbedder
from .trackers import SackMotionTracker, OwnershipMemory


def carry_zone_bounds(person_box, cfg: dict):
    px1, py1, px2, py2 = person_box
    bh = py2 - py1
    bw = px2 - px1
    margin_x = int(bw * 0.25)          # slightly wider horizontally too
    return (
        px1 - margin_x,
        int(py1 + bh * cfg["carry_zone_top"]),   # cfg value now -2.0 covers 2× height above head
        px2 + margin_x,
        int(py1 + bh * cfg["carry_zone_bot"]),
    )


def association_score(sx1, sy1, sx2, sy2, scx, scy,
                      person_box, cfg: dict,
                      sack_emb=None, owner_emb_fn=None, pid=None) -> float:
    px1, py1, px2, py2 = person_box
    bh  = max(1, py2 - py1)
    bw  = max(1, px2 - px1)
    pcx = (px1 + px2) / 2

    # ── Distance score ────────────────────────────────────────
    # For objects above the head (scy < py1) we anchor distance to
    # the top-centre of the person box rather than the centroid.
    # This prevents penalising boxes that are spatially correct
    # (directly above the head) just because they are far from py2/2.
    max_dist = cfg["max_assoc_dist"]
    if scy < py1:
        # above head: measure from top-centre (pcx, py1)
        dist = np.hypot(scx - pcx, scy - py1)
    else:
        pcy  = (py1 + py2) / 2
        dist = np.hypot(scx - pcx, scy - pcy)
    if dist >= max_dist:
        return 0.0
    dist_score = 1.0 - dist / max_dist

    # ── Overlap / carry-zone score ────────────────────────────
    zx1, zy1, zx2, zy2 = carry_zone_bounds(person_box, cfg)
    ix1 = max(sx1, zx1); iy1 = max(sy1, zy1)
    ix2 = min(sx2, zx2); iy2 = min(sy2, zy2)
    inter         = max(0, ix2-ix1) * max(0, iy2-iy1)
    sack_area     = max(1, (sx2-sx1) * (sy2-sy1))
    overlap_score = inter / sack_area

    # ── Height score ──────────────────────────────────────────
    # rel_y < 0  → object is above the person's head  (ideal for head-carry)
    # rel_y = 0  → object at top of person box
    # rel_y = 1  → object at feet
    rel_y = (scy - py1) / bh
    if rel_y < 0:
        # Above the head: reward based on horizontal alignment only.
        # A box directly overhead (scx ≈ pcx) gets full height score.
        horiz_align = max(0.0, 1.0 - abs(scx - pcx) / (bw * 0.8))
        height_score = horiz_align
    else:
        # Below head-top: original formula, penalise as it drops lower
        height_score = max(0.0, 1.0 - rel_y * 1.5)

    # ── Appearance score ──────────────────────────────────────
    app_score = 0.5
    if sack_emb is not None and owner_emb_fn is not None and pid is not None:
        person_emb = owner_emb_fn(pid)
        if person_emb is not None:
            app_score = max(0.0, DeepEmbedder.cosine(sack_emb, person_emb))

    return (cfg["w_distance"]   * dist_score
          + cfg["w_overlap"]    * overlap_score
          + cfg["w_height"]     * height_score
          + cfg["w_appearance"] * app_score)


def assign_sacks_hungarian(
    raw_sacks,
    person_boxes: dict,
    motion_tracker: SackMotionTracker,
    ownership_mem: OwnershipMemory,
    cfg: dict,
    sack_embeddings: dict = None,
    person_emb_fn=None,
    persons_past_gate: set = None,
):
    """
    Hungarian-algorithm assignment with ownership persistence.

    Args:
        raw_sacks:          Raw sack detections for this frame.
        person_boxes:       Dict of pid → (x1, y1, x2, y2).
        motion_tracker:     SackMotionTracker instance.
        ownership_mem:      OwnershipMemory instance.
        cfg:                Configuration dict.
        sack_embeddings:    Optional dict of sid → embedding vector.
        person_emb_fn:      Optional callable pid → embedding (or None).
        persons_past_gate:  (v20) Set of person IDs that have already
                            crossed the gate.  These are excluded from the
                            cost matrix so they cannot be assigned new sack
                            ownership after delivery.  Passing None or an
                            empty set reproduces v19 behaviour.

    Returns:
        sack_owner   dict[sid -> pid]
        sack_scores  dict[sid -> float]   (persisted owner's score)
        person_load  dict[pid -> int]     (current sack count)
        stats        dict                 (raw/still/assigned counts)

    FIX (Critical Bug #1): sack_scores now stores the *persisted owner's*
    score (via ownership_mem.get_score) rather than the raw candidate
    score.  Previously, when ownership_mem.update() resolved to a different
    (already-persisted) pid, sack_scores stored the new candidate's lower
    raw score against the old owner — causing false LOW-CONF anomalies and
    suppressing peak counts.

    v20 FIX (P#457 class): persons in ``persons_past_gate`` are excluded
    from the cost matrix so a bystander who has already been counted cannot
    win Hungarian assignment away from a legitimate carrier.  The caller may
    still pre-filter ``person_boxes`` before passing it in — this guard is
    defence-in-depth, not a replacement for caller filtering.
    """
    _past = persons_past_gate or set()

    # v20: exclude persons already past the gate from assignment candidates
    eligible_boxes = {pid: box for pid, box in person_boxes.items()
                      if pid not in _past}

    person_ids = list(eligible_boxes.keys())
    k          = cfg["max_sacks_per_person"]

    moving_sacks = [(sid, sx1, sy1, sx2, sy2, scx, scy, sc)
                    for (sid, sx1, sy1, sx2, sy2, scx, scy, sc) in raw_sacks
                    if not motion_tracker.is_still(sid)]

    n_still     = len(raw_sacks) - len(moving_sacks)
    sack_owner  = {}
    sack_scores = {}
    person_load = defaultdict(int)

    if not moving_sacks or not person_ids:
        return sack_owner, sack_scores, dict(person_load), {
            "raw": len(raw_sacks), "still": n_still, "assigned": 0
        }

    n_s       = len(moving_sacks)
    n_virtual = len(person_ids) * k
    score_matrix = np.zeros((n_s, n_virtual), dtype=float)

    for si, (sid, sx1, sy1, sx2, sy2, scx, scy, sc) in enumerate(moving_sacks):
        s_emb = (sack_embeddings or {}).get(sid)
        for pi, pid in enumerate(person_ids):
            s = association_score(sx1, sy1, sx2, sy2, scx, scy,
                                  eligible_boxes[pid], cfg,
                                  sack_emb=s_emb,
                                  owner_emb_fn=person_emb_fn,
                                  pid=pid)
            for copy in range(k):
                score_matrix[si, pi * k + copy] = s

    cost_matrix  = -score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    min_score  = cfg.get("min_assoc_score", 0.05)
    n_assigned = 0
    raw_owner  = {}
    raw_scores = {}

    for si, vi in zip(row_ind, col_ind):
        if score_matrix[si, vi] < min_score:
            continue
        sid = moving_sacks[si][0]
        pid = person_ids[vi // k]
        raw_owner[sid]  = pid
        raw_scores[sid] = float(score_matrix[si, vi])
        n_assigned += 1

    # Apply ownership persistence.
    # CRITICAL BUG #1 FIX: use ownership_mem.get_score(sid) — the score
    # that was stored when the *persisted* owner was committed — rather than
    # raw_scores[sid] which belongs to the new candidate (possibly a
    # different person with a lower score).
    for sid, new_pid in raw_owner.items():
        resolved_pid      = ownership_mem.update(sid, new_pid, raw_scores[sid])
        sack_owner[sid]   = resolved_pid
        sack_scores[sid]  = ownership_mem.get_score(sid)   # ← FIX
        person_load[resolved_pid] += 1

    return sack_owner, sack_scores, dict(person_load), {
        "raw": len(raw_sacks), "still": n_still, "assigned": n_assigned
    }
