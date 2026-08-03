"""
exit_landing.py — Landing zone sack tracking for the exit counter.

Responsibilities
----------------
1. Count distinct sack detections inside the landing zone each frame.
2. Track the monotonically increasing peak pile count.
3. When the pile count increases, emit a DIRECT_LAND exit event
   (catches sacks that were thrown too fast for the door crossing detector).
4. Mark sack IDs as "landed" so exit_crossing.py can confirm tentatives.
5. Detect when a sack is picked up (pile count drops + person nearby).

Pile-count approach — REDESIGNED
----------------------------------
Individual sack ID tracking inside the landing zone is unreliable for
PILE FORMATION because:
  - Sacks pile on top of each other (YOLO merges detections)
  - IDs reset when a sack is briefly occluded by another

But individual ID tracking IS reliable for the moment a single sack
first becomes confirmed and still — that detection event doesn't
require the whole pile to be visible at once. The original design
counted "how many sacks are visible THIS EXACT FRAME" and only
advanced when that single-frame count exceeded all previous
single-frame counts. This proved too fragile against normal YOLO
flicker in a crowded pile (frame-to-frame detection counts swinging
2 -> 7 -> 3 purely from occlusion noise), so the true total count
could go unmet for an entire session even when every individual sack
WAS detected and confirmed at some point. This was the root cause of
real footage showing only 1 sack counted out of 5 actually present.

The fix: accumulate into `state["landed_sacks"]` — a SET that only
ever grows across the whole session. Every sack ID that is EVER
confirmed stationary in the zone, at any point, gets added once. The
exit count is the size of this set. A sack doesn't need to be detected
simultaneously with its neighbors in the pile — it just needs its own
few frames of clear, still detection at some point.

  frame N:   sack #3 confirmed still -> landed_sacks={3}            (count=1)
  frame N+k: sack #7 confirmed still -> landed_sacks={3,7}          (count=2)
  frame N+m: sack #3 occluded, undetected -> landed_sacks unchanged (count=2, not decremented)
  frame N+p: sack #12 confirmed still -> landed_sacks={3,7,12}      (count=3)

Discrepancy detection
----------------------
At the end of each session, compare:
  - total_sacks_out (door crossing counter)
  - landing_exit_count (landing zone accumulator)

If they differ by more than 1, a discrepancy flag is added to the log.

Stillness detection — shared with entry mode
----------------------------------------------
Earlier versions of this module used a separate, hand-rolled stillness
check tuned with arbitrary defaults (still_speed=3.0, still_frames=8)
that were never connected to config.yaml. This caused exit-mode
stillness detection to behave differently from entry mode's
ground-sack suppression — even though both are answering the same
underlying question ("has this sack stopped moving?") on the SAME
YOLO model output.

LandingZoneTracker is now a thin wrapper around the real
SackMotionTracker class (sack_counter/trackers.py) — the exact same
class entry mode's ground-sack suppression uses — fed with the exact
same still_speed_thresh / still_speed_window config values from
config.yaml. This means a sack lying still in the landing zone is
judged "still" by the identical logic and identical tuning that
already works correctly for ground sacks in entry mode.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..trackers import SackMotionTracker

if TYPE_CHECKING:
    from .exit_state import ExitPipelineState
    from .landing_zone import LandingZone

logger = logging.getLogger(__name__)

# Fallback defaults — only used if config values are unavailable.
# Matches config.py's DEFAULT_CFG still_speed_thresh / still_speed_window
# so behavior is consistent even without an explicit config load.
_DEFAULT_STILL_SPEED_THRESH  = 1.5
_DEFAULT_STILL_SPEED_WINDOW  = 8
_DEFAULT_STILL_FRAMES        = 8   # consecutive still-classified frames

# How close (pixels) a newly-still sack must be to an EXISTING landed
# sack's position to be treated as the same physical sack re-detected
# under a new track ID, rather than a genuinely new sack. Sack-sized
# objects in this pipeline are roughly 40-80px wide on screen, so this
# default comfortably catches "same sack, ID churned" while still being
# tight enough not to merge two genuinely separate, closely-piled sacks.
_DUPLICATE_RADIUS_PX = 35

# How many frames since a landed sack's position was last refreshed for
# it to still be eligible for spatial dedup matching. Required ALONGSIDE
# proximity — see _find_nearby_landed_sack docstring for why distance
# alone is insufficient. Should comfortably exceed exit_tracker.py's
# miss_frames (default 8) so a sack that briefly drops detection and
# gets evicted, then immediately re-detected under a new ID, is still
# correctly matched against its own recent position.
_DUPLICATE_RECENCY_WINDOW = 20

# Fraction of a sack's bounding box that must fall inside the landing zone
# polygon for it to count as "in the zone".
_DEFAULT_MIN_OVERLAP = 0.30


class LandingZoneTracker:
    """
    Tracks sack motion inside the landing zone across frames.

    Thin wrapper around SackMotionTracker (the same stillness-detection
    class entry mode uses for ground-sack suppression) plus a simple
    consecutive-still-frame counter on top, since "landed" requires not
    just "currently still" but "has been still for N frames running" —
    a sack mid-throw can have a single near-zero-velocity sample by
    chance, and we don't want that to register as landed immediately.

    Grace period for flickering detections
    -----------------------------------------
    A sack in a crowded pile is frequently NOT detected for 1-3
    consecutive frames at a time due to YOLO occlusion/confidence
    flicker (confirmed directly from real diagnostic footage, where
    per-frame detection counts swung from 2 to 7 and back with no
    stable pattern). Naively calling cleanup() every frame with only
    the currently-visible ID set would wipe a sack's accumulated
    still-streak progress on every single flicker, making it almost
    impossible for `still_frames` consecutive still frames to ever
    actually accumulate for a sack in a busy pile.

    `miss_grace_frames` tolerates a sack being briefly ABSENT (e.g.
    occluded behind another sack for a frame or two) WITHOUT resetting
    its motion history or still-streak. Only if a sack is absent for
    longer than the grace period is its tracking state actually cleared.

    Jitter tolerance for DETECTED-but-momentarily-non-still frames
    ------------------------------------------------------------------
    A separate, related problem: even when a sack IS detected every
    single frame (no gaps at all), SackMotionTracker.is_still() can
    occasionally flip False for one frame due to sub-pixel detection
    noise — the bounding box center shifting by a few pixels as YOLO's
    confidence fluctuates, even though the sack hasn't physically moved.
    The original design hard-reset the still-streak to ZERO on any
    single non-still reading, which (combined with real-world detection
    flicker) could make some sacks take many real-world seconds to
    finally accumulate `still_frames` truly CONSECUTIVE still readings,
    even though the sack visibly landed and stopped moving immediately.

    `jitter_tolerance` allows the still-streak to absorb a small number
    of isolated non-still frames WITHOUT a full reset — it decrements
    by 1 instead of resetting to 0, as long as the streak hasn't
    accumulated more than `jitter_tolerance` consecutive non-still
    readings in a row. Sustained real motion (more than
    `jitter_tolerance` consecutive non-still frames) still fully resets
    progress, so this does not weaken the actual stillness requirement
    for a sack that is genuinely moving.

    Args:
        still_speed_thresh: Pixels/frame below which a sack is considered
            stationary for a given frame.  Should match config.yaml's
            still_speed_thresh (same value entry mode uses).
        still_speed_window:  Number of recent positions SackMotionTracker
            averages over to decide "is_still" for the current frame.
        still_frames:        Consecutive is_still()==True frames required
            before a sack is confirmed "landed" (not just momentarily slow).
        miss_grace_frames:   Frames a sack may be briefly undetected
            before its tracking state is actually cleared. Protects
            against single-frame flicker resetting progress.
        jitter_tolerance:    Consecutive non-still readings (while still
            being detected every frame) allowed before a full
            still-streak reset. Each isolated non-still reading
            decrements the streak by 1 instead of zeroing it, as long
            as this tolerance hasn't been exceeded.
    """

    def __init__(
        self,
        still_speed_thresh: float = _DEFAULT_STILL_SPEED_THRESH,
        still_speed_window:  int   = _DEFAULT_STILL_SPEED_WINDOW,
        still_frames:        int   = _DEFAULT_STILL_FRAMES,
        miss_grace_frames:   int   = 5,
        jitter_tolerance:    int   = 2,
    ):
        self.still_frames      = still_frames
        self.miss_grace_frames = miss_grace_frames
        self.jitter_tolerance  = jitter_tolerance
        self._motion = SackMotionTracker(
            speed_thresh=still_speed_thresh,
            speed_window=still_speed_window,
        )
        # sid -> consecutive frames where SackMotionTracker.is_still() == True
        self._still_streak: dict[int, int] = {}
        # sid -> consecutive frames since last detected (reset to 0 on update)
        self._miss_streak: dict[int, int] = {}
        # sid -> consecutive non-still readings while still being
        # detected (separate from miss_streak — this counts jitter,
        # not absence). Reset to 0 whenever is_still() is True.
        self._jitter_streak: dict[int, int] = {}

    def update(self, sid: int, cx: int, cy: int) -> None:
        """Record a new position for sack *sid*."""
        self._motion.update(sid, cx, cy)
        self._miss_streak[sid] = 0

        if self._motion.is_still(sid):
            self._still_streak[sid] = self._still_streak.get(sid, 0) + 1
            self._jitter_streak[sid] = 0
        else:
            jitter = self._jitter_streak.get(sid, 0) + 1
            self._jitter_streak[sid] = jitter
            if jitter <= self.jitter_tolerance:
                # Isolated non-still reading — absorb it with a small
                # decrement instead of a full reset.
                self._still_streak[sid] = max(
                    0, self._still_streak.get(sid, 0) - 1
                )
            else:
                # Sustained motion beyond the jitter tolerance — this is
                # genuine movement, not sensor noise. Full reset.
                self._still_streak[sid] = 0

    def is_still(self, sid: int) -> bool:
        """
        Return True if sack *sid* has been classified stationary by
        SackMotionTracker for at least ``still_frames`` consecutive frames.
        """
        return self._still_streak.get(sid, 0) >= self.still_frames

    def last_position(self, sid: int) -> tuple:
        """
        Return the most recent recorded (cx, cy) for sack *sid*, or
        (None, None) if no position has been recorded yet.

        Used for spatial deduplication — when a sack is about to be
        marked newly landed, its last known position is checked against
        already-landed sacks to detect "same physical sack, new track
        ID" rather than letting a genuinely new sid auto-count.
        """
        hist = self._motion.history.get(sid)
        if not hist:
            return (None, None)
        return tuple(hist[-1])

    def cleanup(self, active_ids: set) -> None:
        """
        Increment miss-streaks for sacks not in *active_ids* this frame.
        Only clear a sack's tracking state once its miss-streak exceeds
        miss_grace_frames — a brief 1-2 frame detection flicker does NOT
        reset accumulated still-streak progress.
        """
        all_tracked = set(self._still_streak.keys()) | set(self._miss_streak.keys())
        for sid in all_tracked:
            if sid in active_ids:
                continue  # update() already reset its miss_streak to 0
            self._miss_streak[sid] = self._miss_streak.get(sid, 0) + 1
            if self._miss_streak[sid] > self.miss_grace_frames:
                self._motion.history.pop(sid, None)
                self._still_streak.pop(sid, None)
                self._miss_streak.pop(sid, None)
                self._jitter_streak.pop(sid, None)


def make_landing_zone_tracker(cfg: dict) -> "LandingZoneTracker":
    """
    Build a LandingZoneTracker using the SAME still-detection config
    values as entry mode's ground-sack suppression, so stillness is
    judged identically in both modes.

    Args:
        cfg: Loaded config dict (from sack_counter.config.load_config).

    Returns:
        Configured LandingZoneTracker.
    """
    return LandingZoneTracker(
        still_speed_thresh=float(cfg.get(
            "still_speed_thresh", _DEFAULT_STILL_SPEED_THRESH)),
        still_speed_window=int(cfg.get(
            "still_speed_window", _DEFAULT_STILL_SPEED_WINDOW)),
        still_frames=int(cfg.get(
            "exit_still_frames", _DEFAULT_STILL_FRAMES)),
        miss_grace_frames=int(cfg.get(
            "exit_landing_miss_grace_frames", 5)),
        jitter_tolerance=int(cfg.get(
            "exit_landing_jitter_tolerance", 2)),
    )


def _find_nearby_landed_sack(
    state: "ExitPipelineState",
    cx: int,
    cy: int,
    exclude_sid: int,
    radius_px: float,
    fn: int,
    recency_window: int,
    currently_active_ids: set,
) -> int | None:
    """
    Check whether any ALREADY-landed sack has a recorded position within
    *radius_px* of (cx, cy), was last seen within *recency_window*
    frames of *fn*, AND is no longer being actively tracked under its
    own ID this frame.

    Used to detect "same physical sack, new track ID" — when YOLO/
    ByteTrack assigns a fresh ID to a sack that never actually moved
    (e.g. after a long stillness period where detection briefly dropped
    out and the old track was evicted), this prevents the new ID from
    being counted as a genuinely new exit.

    ALL THREE conditions are required — proximity, recency, AND the old
    ID being currently absent:

    - Proximity alone is not enough: two genuinely separate sacks
      sitting close together in a tight pile (closer than radius_px
      apart) would otherwise be incorrectly merged, even when both are
      live, independently tracked sacks.
    - The "currently absent" check is what actually distinguishes ID
      churn from a tight pile: if the candidate sid is STILL being
      detected this frame under its own ID, the new sid cannot
      possibly be the same physical sack — they coexist simultaneously,
      so they must be two different sacks. Only an ID that has
      genuinely disappeared (evicted, no longer in this frame's
      detections) is eligible to be "resurrected" under a new sid.

    Args:
        state:                 ExitPipelineState.
        cx, cy:                Position to check.
        exclude_sid:           The sid being checked — never matches itself.
        radius_px:             Maximum distance to consider "the same sack".
        fn:                    Current frame number.
        recency_window:        Maximum frames since the candidate's last
                               update for it to still be eligible.
        currently_active_ids:  Sack IDs detected in the landing zone
                               THIS frame. A candidate sid present in
                               this set is still alive under its own ID
                               and can never be matched as a duplicate.

    Returns:
        The sid of the nearby, recently-vacated existing landed sack,
        or None if no such sack is found.
    """
    positions  = state["landed_positions"]
    last_seen  = state["landed_last_seen_frame"]
    for other_sid, (ox, oy) in positions.items():
        if other_sid == exclude_sid:
            continue
        if other_sid in currently_active_ids:
            # Still alive under its own ID this very frame — cannot be
            # the same physical sack as a DIFFERENT sid also present
            # this frame. Two coexisting detections are two sacks.
            continue
        age = fn - last_seen.get(other_sid, fn)
        if age > recency_window:
            continue  # too long ago — likely a genuinely different sack
        dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
        if dist <= radius_px:
            return other_sid
    return None


def update_landing_zone(
    state: "ExitPipelineState",
    landing_zone: "LandingZone",
    sack_detections: list[tuple],
    fn: int,
    motion_tracker: "LandingZoneTracker",
    dedup_radius_px: float = _DUPLICATE_RADIUS_PX,
    dedup_recency_window: int = _DUPLICATE_RECENCY_WINDOW,
    min_overlap: float = _DEFAULT_MIN_OVERLAP,
) -> list[dict]:
    """
    Update landing zone state and emit DIRECT_LAND events when new sacks land.

    REDESIGNED counting strategy
    -----------------------------
    Count is driven by the size of `landed_sacks` — a SET that only
    ever grows, accumulating every sack ID that has EVER been confirmed
    stationary in the zone across the whole session — rather than the
    previous fragile approach of counting how many sacks YOLO happened
    to detect simultaneously in a single frame. See exit_landing.py
    module docstring for the full explanation.

    Args:
        state:           ExitPipelineState (modified in-place).
        landing_zone:    Calibrated LandingZone polygon.
        sack_detections: List of (sid, x1, y1, x2, y2, cx, cy, conf) tuples
                         for ALL sacks detected this frame (not just confirmed).
        fn:              Current frame number.
        motion_tracker:  LandingZoneTracker instance (shared across calls).
                         Owns its own stillness threshold internally —
                         build it once via make_landing_zone_tracker(cfg).

    Returns:
        List of new DIRECT_LAND event dicts (may be empty).
    """
    new_events: list[dict] = []

    # ── Find sacks sufficiently overlapping the landing zone ──
    # Uses fractional bbox overlap (not just centroid) so sacks that are
    # mostly inside the zone but straddling its edge — very common at
    # the boundary of a pile — are still correctly detected.
    in_zone_ids: set[int] = set()
    for (sid, x1, y1, x2, y2, cx, cy, conf) in sack_detections:
        if landing_zone.bbox_overlaps(x1, y1, x2, y2, min_fraction=min_overlap):
            in_zone_ids.add(sid)
            motion_tracker.update(sid, cx, cy)

    # ── Mark new still sacks as landed — THIS is the accumulator ──
    # state["landed_sacks"] only ever grows across the whole session.
    # A sack just needs to be confirmed still ONCE, at any point — it
    # does NOT need to be simultaneously visible alongside every other
    # sack currently in the pile. This replaces the old approach of
    # counting len(in_zone_ids) per frame, which was too fragile:
    # YOLO detection in a crowded pile flickers frame to frame
    # (occlusion, brief confidence dips), so "all N sacks visible at
    # once in one single frame" could go unmet for an entire session
    # even when every individual sack WAS detected and confirmed at
    # some point. This was the root cause of "only counted 1 out of 5
    # real sacks".
    #
    # SPATIAL DEDUPLICATION (added after real-world testing showed a
    # single stationary sack counted 2-3 times): the ID-only check
    # above protects against the SAME track ID re-triggering, but does
    # nothing if YOLO/ByteTrack assigns a NEW ID to a sack that never
    # actually moved — which happens often for a sack untouched for a
    # long time, where confidence/detection briefly drops out (>8
    # missed frames triggers eviction in exit_tracker.py) and a fresh
    # ID is assigned on re-detection. Before adding a sid as newly
    # landed, check whether an EXISTING landed sack already occupies
    # roughly the same position — if so, treat this as the same
    # physical sack re-appearing under a new ID, not a new exit.
    newly_landed: list[int] = []
    for sid in in_zone_ids:
        if sid in state["landed_sacks"]:
            # Already registered — refresh its last-seen frame so the
            # recency window for dedup matching stays current as long
            # as this sack remains genuinely visible.
            cx, cy = motion_tracker.last_position(sid)
            if cx is not None:
                state["landed_positions"][sid] = (cx, cy)
                state["landed_last_seen_frame"][sid] = fn
            continue  # already registered — don't double count

        if not motion_tracker.is_still(sid):
            continue

        cx, cy = motion_tracker.last_position(sid)
        if cx is None:
            continue

        duplicate_of = _find_nearby_landed_sack(
            state, cx, cy, exclude_sid=sid,
            radius_px=dedup_radius_px,
            fn=fn, recency_window=dedup_recency_window,
            currently_active_ids=in_zone_ids,
        )
        if duplicate_of is not None:
            # Same physical sack, new track ID — alias it so future
            # lookups by this sid are also recognized as already-landed,
            # without inflating the count. Also inherit the position/
            # recency tracking under the NEW sid so subsequent frames
            # keep the recency window alive correctly.
            state["landed_sacks"].add(sid)
            state["landed_aliases"].add(sid)
            state["landed_positions"][sid] = (cx, cy)
            state["landed_last_seen_frame"][sid] = fn
            logger.debug(
                "LANDED sack#%d  DEDUPED as same physical sack as #%d  "
                "(within %dpx, %d frames recency)  frame=%d",
                sid, duplicate_of, dedup_radius_px,
                dedup_recency_window, fn,
            )
            continue

        state["landed_sacks"].add(sid)
        state["landed_positions"][sid] = (cx, cy)
        state["landed_last_seen_frame"][sid] = fn
        newly_landed.append(sid)
        logger.debug("LANDED sack#%d  frame=%d", sid, fn)

    # ── Exit count = size of the accumulated landed_sacks set ─────
    # Every newly landed sack this frame is one more confirmed exit,
    # full stop — no need to wait for the whole pile to be visible
    # simultaneously in a single frame.
    if newly_landed:
        delta = len(newly_landed)
        state["landing_exit_count"] = state["landing_exit_count"] + delta
        # Count DISTINCT physical sacks: landed_sacks also holds the alias
        # IDs recorded when a sack's track ID churned, so using its raw
        # length made the HUD's "PILE NOW" drift above the actual count by
        # one per dedup — reporting more sacks on the floor than were
        # counted as having exited.
        state["landing_peak_count"] = (
            len(state["landed_sacks"]) - len(state["landed_aliases"])
        )

        for sid in newly_landed:
            event = {
                "frame":      fn,
                "sack_id":    sid,
                "type":       "direct_land",
                "trigger":    "landing_zone_accumulated",
                "pile_count": state["landing_peak_count"],

            }
            new_events.append(event)
            state["exit_log"].append(event)

        logger.info(
            "LANDING-ACCUMULATE  newly_landed=%s  total_landed=%d  "
            "landing_exits=%d  frame=%d",
            newly_landed, state["landing_peak_count"],
            state["landing_exit_count"], fn,
        )

    # ── Cleanup motion tracker for sacks no longer visible ────
    motion_tracker.cleanup(in_zone_ids)

    return new_events


def check_discrepancy(
    state: "ExitPipelineState",
    fn: int,
    tolerance: int = 1,
) -> bool:
    """
    Compare door crossing count against landing zone count.
    Flag a discrepancy if they differ by more than *tolerance*.

    Args:
        state:     ExitPipelineState.
        fn:        Current frame number.
        tolerance: Acceptable difference before flagging.

    Returns:
        True if a discrepancy was detected and flagged.
    """
    door_count    = state["total_sacks_out"]
    landing_count = state["landing_exit_count"]
    diff = abs(door_count - landing_count)

    if diff > tolerance:
        flag = {
            "frame":          fn,
            "door_count":     door_count,
            "landing_count":  landing_count,
            "diff":           diff,
        }
        state["discrepancy_flags"].append(flag)
        logger.warning(
            "DISCREPANCY  door=%d  landing=%d  diff=%d  frame=%d",
            door_count, landing_count, diff, fn,
        )
        return True

    return False


def reconcile_counts(state: "ExitPipelineState") -> int:
    """
    Return the best estimate of total sacks exited.

    Strategy: take the maximum of door_count and landing_count.
    The door crossing may miss fast throws; the landing zone may
    under-count when sacks pile up and YOLO merges them.
    Taking the max gives the most conservative (highest) reliable count.

    Args:
        state: ExitPipelineState.

    Returns:
        Best estimate of total sacks out.
    """
    return max(state["total_sacks_out"], state["landing_exit_count"])
