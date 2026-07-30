"""
state_machine.py — Per-sack lifecycle state machine for Sack Counter v20.

Each sack tracked by the system transitions through well-defined states
instead of being governed by scattered ``if/else`` checks.

States
------
ON_GROUND   : Sack is stationary on the floor; excluded from counting.
DETECTED    : Newly detected; not yet confirmed.
CONFIRMED   : Seen for enough frames to trust.
PICKED_UP   : Associated with a person after being on the ground.
CARRIED     : Assigned to a person and moving toward the gate.
APPROACHING : Inside the gate approach window.
DELIVERED   : Has crossed gate B; delivery committed.
LOST        : Track dropped; may be recovered via ghost or ReID.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class SackState(Enum):
    ON_GROUND   = auto()
    DETECTED    = auto()
    CONFIRMED   = auto()
    PICKED_UP   = auto()
    CARRIED     = auto()
    APPROACHING = auto()
    DELIVERED   = auto()
    LOST        = auto()


# Valid one-step transitions
_TRANSITIONS: dict[SackState, set[SackState]] = {
    SackState.ON_GROUND:   {SackState.DETECTED, SackState.PICKED_UP, SackState.LOST},
    SackState.DETECTED:    {SackState.CONFIRMED, SackState.ON_GROUND, SackState.LOST},
    SackState.CONFIRMED:   {SackState.CARRIED, SackState.ON_GROUND, SackState.LOST},
    SackState.PICKED_UP:   {SackState.CARRIED, SackState.ON_GROUND, SackState.LOST},
    SackState.CARRIED:     {SackState.APPROACHING, SackState.ON_GROUND,
                            SackState.CONFIRMED, SackState.LOST},
    SackState.APPROACHING: {SackState.DELIVERED, SackState.CARRIED, SackState.LOST},
    SackState.DELIVERED:   set(),                 # terminal
    SackState.LOST:        {SackState.DETECTED},  # re-entry after ReID
}


@dataclass
class SackRecord:
    """
    Complete lifecycle record for a single tracked sack.

    Attributes:
        sack_id:        YOLO tracker ID.
        state:          Current lifecycle state.
        owner_pid:      Current assigned person ID (None if unowned).
        first_seen:     Frame number when first detected.
        last_seen:      Frame number of last detection.
        history:        List of (frame, SackState) transition log.
        position:       Last known (cx, cy) centroid.
        ground_since:   Frame when sack became stationary (for suppression).
        delivered_at:   Frame when DELIVERED state was entered (for safe pruning).
    """
    sack_id:      int
    state:        SackState   = SackState.DETECTED
    owner_pid:    Optional[int] = None
    first_seen:   int         = 0
    last_seen:    int         = 0
    history:      list        = field(default_factory=list)
    position:     tuple[int, int] = (0, 0)
    ground_since: Optional[int]   = None
    delivered_at: Optional[int]   = None

    def transition(self, new_state: SackState, frame_no: int) -> bool:
        """
        Attempt a state transition.

        Args:
            new_state: Target state.
            frame_no:  Current frame number (logged).

        Returns:
            True if the transition was accepted, False if invalid.
        """
        if new_state not in _TRANSITIONS.get(self.state, set()):
            return False
        self.history.append((frame_no, self.state, new_state))
        self.state = new_state
        return True

    def force_transition(self, new_state: SackState, frame_no: int) -> None:
        """
        Unconditionally set state (use sparingly — bypasses guard).

        Useful for initialisation or recovery after edge cases.
        """
        self.history.append((frame_no, self.state, new_state))
        self.state = new_state


class SackStateMachineRegistry:
    """
    Registry that owns a SackRecord for every active tracked sack.

    The main pipeline calls :meth:`get_or_create`, :meth:`transition`,
    and :meth:`cleanup` each frame; the state machine handles the rest.
    """

    def __init__(self) -> None:
        self._records: dict[int, SackRecord] = {}

    # ── Public API ────────────────────────────────────────────────

    def get_or_create(self, sack_id: int, frame_no: int) -> SackRecord:
        """
        Return the existing SackRecord for *sack_id*, or create one.

        Args:
            sack_id:  YOLO tracker ID.
            frame_no: Current frame number.

        Returns:
            The SackRecord for *sack_id*.
        """
        if sack_id not in self._records:
            self._records[sack_id] = SackRecord(
                sack_id=sack_id,
                first_seen=frame_no,
                last_seen=frame_no,
            )
        return self._records[sack_id]

    def get(self, sack_id: int) -> Optional[SackRecord]:
        """Return existing record or None."""
        return self._records.get(sack_id)

    def transition(self, sack_id: int, new_state: SackState,
                   frame_no: int) -> bool:
        """
        Transition *sack_id* to *new_state*.

        Args:
            sack_id:   YOLO tracker ID.
            new_state: Target state.
            frame_no:  Current frame number.

        Returns:
            True if accepted; False if sack unknown or transition invalid.
        """
        rec = self._records.get(sack_id)
        if rec is None:
            return False
        return rec.transition(new_state, frame_no)

    def mark_delivered(self, sack_id: int, frame_no: int) -> None:
        """
        Force *sack_id* into DELIVERED state regardless of current state.

        Args:
            sack_id:  YOLO tracker ID.
            frame_no: Current frame number.
        """
        rec = self._records.get(sack_id)
        if rec:
            rec.force_transition(SackState.DELIVERED, frame_no)
            rec.delivered_at = frame_no

    # Minimum frames to keep a DELIVERED record before pruning.
    # Must be long enough that ByteTrack cannot recycle the same tracker ID
    # for a new sack within this window (typically safe at 60+ frames).
    _DELIVERED_KEEP_FRAMES: int = 90

    def cleanup(self, active_sids: set[int], frame_no: int) -> None:
        """
        Mark sacks absent from *active_sids* as LOST and prune old DELIVERED.

        BUG #9 FIX: DELIVERED records are now kept for at least
        ``_DELIVERED_KEEP_FRAMES`` frames before deletion.  Previously they
        were pruned immediately on the first frame the tracker ID was absent,
        which meant ByteTrack could recycle the same ID for a new sack and
        the registry would silently resurrect it in DETECTED state — losing
        the DELIVERED history and risking a double-count.

        Args:
            active_sids: Set of sack IDs seen this frame.
            frame_no:    Current frame number.
        """
        for sid, rec in list(self._records.items()):
            if sid not in active_sids:
                if rec.state not in (SackState.DELIVERED, SackState.LOST):
                    rec.transition(SackState.LOST, frame_no)
                # Prune DELIVERED records only after the keep window expires
                if (rec.state == SackState.DELIVERED
                        and rec.delivered_at is not None
                        and frame_no - rec.delivered_at >= self._DELIVERED_KEEP_FRAMES):
                    del self._records[sid]

    def in_state(self, state: SackState) -> list[int]:
        """Return list of sack IDs currently in *state*."""
        return [sid for sid, rec in self._records.items()
                if rec.state == state]

    def all_records(self) -> dict[int, SackRecord]:
        """Return a read-only view of all records."""
        return self._records
