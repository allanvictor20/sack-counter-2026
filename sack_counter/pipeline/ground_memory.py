"""
ground_memory.py — Ground-object memory for Sack Counter v20.

Tracks sacks and boxes that have been resting on the floor so the pipeline
can distinguish:

    1. A sack being carried into the scene  (NEW — should be counted)
    2. A sack that was already on the ground (KNOWN — ignore until picked up)

This avoids double-counting sacks that appear stationary near the gate
before a worker picks them up.

Classes
-------
GroundObject   : A single recorded floor sack/box.
GroundMemory   : Registry that stores and queries ground objects.

v20 changes
-----------
* ``cleanup()`` now prunes grounded objects that have not been seen for
  more than ``stale_frames`` frames, preventing unbounded memory growth
  on long-running sessions where sacks disappear (tracker dies) without
  ever being picked up.
* ``_cosine`` helper removed — callers use ``DeepEmbedder.cosine`` directly
  to avoid maintaining a local duplicate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..embedder import DeepEmbedder


@dataclass
class GroundObject:
    """
    A tracked object known to have been on the ground.

    Attributes:
        obj_id:       Tracker ID.
        position:     Last known (cx, cy) centroid.
        appearance:   Embedding vector when object was first grounded.
        first_seen:   Frame when first classified as ground object.
        last_seen:    Frame of most recent still observation.
        picked_up:    True once the object starts moving again.
        pickup_frame: Frame when pickup was detected.
    """
    obj_id:       int
    position:     tuple[int, int]
    appearance:   Optional[np.ndarray] = None
    first_seen:   int                  = 0
    last_seen:    int                  = 0
    picked_up:    bool                 = False
    pickup_frame: Optional[int]        = None


class GroundMemory:
    """
    Registry of objects known to be resting on the floor.

    The pipeline calls :meth:`mark_grounded` when a sack becomes still
    and :meth:`mark_picked_up` when it starts moving again.
    :meth:`is_known_ground_object` lets the assignment layer skip sacks
    that haven't been picked up yet.

    Args:
        appearance_threshold: Cosine similarity above which two appearance
            vectors are considered the same object (used for lost+reappeared
            ground sacks).
        position_threshold_px: Maximum centroid distance to match a
            reappearing object to a known ground object.
        pickup_stable_frames: Number of consecutive moving frames before a
            pickup is confirmed (mirrors ``lift_owner_stable_frames``).
        stale_frames: Number of frames after ``last_seen`` before a grounded
            (not-yet-picked-up) record is pruned.  Prevents unbounded memory
            growth when the tracker drops a ground sack without recovering it.
            Default 300 (~15 s at 20 fps).
    """

    def __init__(
        self,
        appearance_threshold:  float = 0.80,
        position_threshold_px: int   = 60,
        pickup_stable_frames:  int   = 6,
        stale_frames:          int   = 300,
    ) -> None:
        self._threshold_app  = appearance_threshold
        self._threshold_pos  = position_threshold_px
        self._pickup_stable  = pickup_stable_frames
        self._stale_frames   = stale_frames
        self._objects:       dict[int, GroundObject] = {}
        self._moving_streak: dict[int, int]          = {}   # sid -> consecutive moving frames

    # ── Lifecycle methods ─────────────────────────────────────

    def mark_grounded(
        self,
        obj_id:     int,
        position:   tuple[int, int],
        frame_no:   int,
        appearance: Optional[np.ndarray] = None,
    ) -> None:
        """
        Record that *obj_id* is stationary on the ground.

        If the object is already registered, updates its last_seen and
        position. Otherwise creates a new GroundObject record.

        Args:
            obj_id:     Tracker ID.
            position:   Current centroid (cx, cy).
            frame_no:   Current frame number.
            appearance: Optional embedding for identity matching.
        """
        if obj_id in self._objects:
            rec = self._objects[obj_id]
            rec.position  = position
            rec.last_seen = frame_no
            rec.picked_up = False
        else:
            self._objects[obj_id] = GroundObject(
                obj_id=obj_id,
                position=position,
                appearance=appearance,
                first_seen=frame_no,
                last_seen=frame_no,
            )
        # Reset movement streak — it's still
        self._moving_streak.pop(obj_id, None)

    def mark_moving(self, obj_id: int, frame_no: int) -> bool:
        """
        Notify that *obj_id* is moving this frame.

        Increments an internal consecutive-frames counter.  Once the
        counter reaches ``pickup_stable_frames`` the object is confirmed
        picked up and removed from the ground registry.

        Args:
            obj_id:   Tracker ID.
            frame_no: Current frame number.

        Returns:
            True if this call confirmed a pickup (streak threshold reached).
        """
        if obj_id not in self._objects:
            return False
        streak = self._moving_streak.get(obj_id, 0) + 1
        self._moving_streak[obj_id] = streak
        if streak >= self._pickup_stable:
            self._confirm_pickup(obj_id, frame_no)
            return True
        return False

    def is_known_ground_object(self, obj_id: int) -> bool:
        """
        Return True if *obj_id* is known to be resting on the ground
        and has not yet been confirmed as picked up.

        Args:
            obj_id: Tracker ID.

        Returns:
            True → exclude from counting; False → count normally.
        """
        rec = self._objects.get(obj_id)
        return rec is not None and not rec.picked_up

    def find_by_position(
        self,
        position: tuple[int, int],
        appearance: Optional[np.ndarray] = None,
    ) -> Optional[int]:
        """
        Find a ground object near *position* (for lost-track recovery).

        Matches by position proximity first; if *appearance* is provided
        also requires cosine similarity above threshold.

        Args:
            position:   Query centroid (cx, cy).
            appearance: Optional embedding for appearance matching.

        Returns:
            Matching tracker ID, or None.
        """
        best_id:   Optional[int]   = None
        best_dist: float           = float("inf")

        for obj_id, rec in self._objects.items():
            if rec.picked_up:
                continue
            dist = float(np.hypot(
                position[0] - rec.position[0],
                position[1] - rec.position[1],
            ))
            if dist >= self._threshold_pos:
                continue
            if appearance is not None and rec.appearance is not None:
                sim = DeepEmbedder.cosine(appearance, rec.appearance)
                if sim < self._threshold_app:
                    continue
            if dist < best_dist:
                best_dist = dist
                best_id   = obj_id

        return best_id

    def cleanup(self, active_ids: set[int], frame_no: int = 0) -> None:
        """
        Prune stale and picked-up records.

        Two pruning rules:
        1. Picked-up objects that are no longer tracked are removed
           immediately (unchanged from v19).
        2. **(v20 new)** Grounded objects not seen for more than
           ``stale_frames`` frames are pruned, preventing unbounded memory
           growth when a tracker dies without the sack ever being picked up.

        Args:
            active_ids: Set of tracker IDs seen this frame.
            frame_no:   Current frame number (needed for stale-TTL pruning).
        """
        for oid in list(self._objects):
            rec = self._objects[oid]
            # Rule 1: confirmed-pickup records whose tracker is gone
            if rec.picked_up and oid not in active_ids:
                del self._objects[oid]
                self._moving_streak.pop(oid, None)
                continue
            # Rule 2: grounded records that have gone stale
            if (not rec.picked_up
                    and oid not in active_ids
                    and frame_no > 0
                    and frame_no - rec.last_seen > self._stale_frames):
                del self._objects[oid]
                self._moving_streak.pop(oid, None)

    # ── Properties ────────────────────────────────────────────

    @property
    def grounded_ids(self) -> set[int]:
        """Set of IDs currently classified as ground objects."""
        return {oid for oid, rec in self._objects.items()
                if not rec.picked_up}

    @property
    def total_grounded(self) -> int:
        """Current count of ground objects (not yet picked up)."""
        return len(self.grounded_ids)

    # ── Private helpers ───────────────────────────────────────

    def _confirm_pickup(self, obj_id: int, frame_no: int) -> None:
        rec = self._objects[obj_id]
        rec.picked_up    = True
        rec.pickup_frame = frame_no
        self._moving_streak.pop(obj_id, None)
