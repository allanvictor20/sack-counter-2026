"""
state.py — Typed state dataclasses for Sack Counter v20.

Replaces the flat 25-key ``dict`` that ``PipelineSession.__init__`` used to
build inline.  All fields keep their original names so existing call-sites
(``session.state["sack_owner"]`` etc.) continue to work via
``PipelineState.__getitem__`` / ``__setitem__`` / ``__contains__`` /
``.get()``, which delegate to the underlying dataclass fields.

Sub-groups
----------
PersonState      — per-person tracking counters, boxes, embeddings
SackStateGroup   — per-sack tracking counters, positions, boxes, embeddings
GroundGroup      — still-sack suppression
DeliveryState    — counts, logs, anomalies
PeakState        — rolling peak-count machinery
CarrierGuardState— guard flags added during v16-v19 patches (renamed from
                   AssignBugState — the fields are first-class carrier
                   management concepts, not temporary workarounds)
PipelineState    — top-level container; also exposes a dict-like interface

v20 changes
-----------
* Removed dead ``PipelineState._resolve()`` method — it was never called;
  ``__getitem__`` and ``__setitem__`` both inlined the same logic.
* Renamed ``AssignBugState`` → ``CarrierGuardState`` to reflect that
  ``sack_carrier_stamp``, ``just_evicted_sacks``, and ``orphaned_peaks``
  are first-class carrier management fields, not ad-hoc patches.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


# ── Sub-group dataclasses ─────────────────────────────────────────────────────

@dataclass
class PersonState:
    """Per-person tracking state."""
    person_boxes:      dict = field(default_factory=dict)          # pid -> (x1,y1,x2,y2)
    person_hit:        Any  = field(default_factory=lambda: defaultdict(int))
    person_miss:       Any  = field(default_factory=lambda: defaultdict(int))
    confirmed_persons: set  = field(default_factory=set)
    person_velocity:   dict = field(default_factory=dict)          # pid -> (vx, vy)
    person_prev_cx:    dict = field(default_factory=dict)          # pid -> int
    person_embeddings: dict = field(default_factory=dict)
    persons_past_gate: set  = field(default_factory=set)
    person_position_history: dict = field(default_factory=dict)
        # pid -> deque of (cx, cy) — used by door_crossing direction confirmation


@dataclass
class SackStateGroup:
    """Per-sack tracking state."""
    raw_sacks:      list = field(default_factory=list)
    raw_boxes:      list = field(default_factory=list)
    sack_hit:       Any  = field(default_factory=lambda: defaultdict(int))
    sack_miss:      Any  = field(default_factory=lambda: defaultdict(int))
    confirmed_sacks: set = field(default_factory=set)
    sack_positions:  dict = field(default_factory=dict)   # sid -> (cx, cy)
    sack_boxes_dict: dict = field(default_factory=dict)   # sid -> (x1,y1,x2,y2)
    sack_owner:      dict = field(default_factory=dict)   # sid -> pid
    sack_scores:     dict = field(default_factory=dict)   # sid -> float
    sack_embeddings: dict = field(default_factory=dict)


@dataclass
class GroundGroup:
    """Ground-sack suppression state."""
    sack_still_frames: dict = field(default_factory=dict)   # sid -> int
    ground_sacks:      set  = field(default_factory=set)
    ground_boxes:      set  = field(default_factory=set)


@dataclass
class DeliveryState:
    """Delivery tracking, logs, and anomalies."""
    person_sack_delivered:  Any  = field(default_factory=lambda: defaultdict(set))
    person_peak_delivery:   dict = field(default_factory=dict)   # pid -> peak int
    total_sacks_counted:    int  = 0
    delivery_log:           list = field(default_factory=list)
    anomaly_log:            list = field(default_factory=list)
    n_anomalies:            int  = 0


@dataclass
class PeakState:
    """Peak-count approach-window machinery."""
    person_peak_count:  Any  = field(default_factory=lambda: defaultdict(int))
    person_peak_used:   Any  = field(default_factory=lambda: defaultdict(lambda: False))
    person_approach_fn: dict = field(default_factory=dict)   # pid -> first approach frame
    person_load:        dict = field(default_factory=dict)   # pid -> current sack count


@dataclass
class CarrierGuardState:
    """
    First-class carrier management state.

    These fields were introduced incrementally in v16-v19 to handle edge
    cases in sack ownership and gate crossing, but they are now stable
    enough to be treated as domain concepts rather than patches.

    Fields
    ------
    just_evicted_sacks : Sack IDs whose ownership was cleared this frame
        (BUG1-A guard — prevents eviction from inflating a new carrier's
        peak count in the same frame).
    sack_carrier_stamp : Maps sack_id → person_id for the first carrier
        that reached the stamp score threshold.  Prevents re-assignment
        until the stamp is cleared on gate crossing or re-entry.
    orphaned_peaks     : Peak snapshot for a person who timed out before
        their sack crossed the gate.  Keyed by person_id; each entry is
        ``{"peak": int, "expires_at": int}``.
    orphaned_sack_owner: Maps sack_id → person_id for sacks that were
        owned by a timed-out person, so the gate-crossing path can still
        commit the correct peak.
    """
    just_evicted_sacks:  set  = field(default_factory=set)
    delivered_sacks: set = field(default_factory=set)
    sack_carrier_stamp:  dict = field(default_factory=dict)   # sid -> pid
    orphaned_peaks:      dict = field(default_factory=dict)   # pid -> {peak, expires_at}
    orphaned_sack_owner: dict = field(default_factory=dict)   # sid -> pid


@dataclass
class DoorState:
    """Door-zone tracking state (v21 — replaces gate crossing state)."""
    door_candidates:   dict = field(default_factory=dict)
    # pid -> DoorCandidateState
    persons_past_door: set  = field(default_factory=set)
    # pids that have been committed this session


# ── Flat-key map: logical name → (group_attr, field_attr) ────────────────────

_FIELD_MAP: dict[str, tuple[str, str]] = {
    # person
    "person_boxes":            ("persons",    "person_boxes"),
    "person_hit":              ("persons",    "person_hit"),
    "person_miss":             ("persons",    "person_miss"),
    "confirmed_persons":       ("persons",    "confirmed_persons"),
    "person_velocity":         ("persons",    "person_velocity"),
    "person_prev_cx":          ("persons",    "person_prev_cx"),
    "person_embeddings":       ("persons",    "person_embeddings"),
    "persons_past_gate":       ("persons",    "persons_past_gate"),
    # sack
    "raw_sacks":               ("sacks",      "raw_sacks"),
    "raw_boxes":               ("sacks",      "raw_boxes"),
    "sack_hit":                ("sacks",      "sack_hit"),
    "sack_miss":               ("sacks",      "sack_miss"),
    "confirmed_sacks":         ("sacks",      "confirmed_sacks"),
    "sack_positions":          ("sacks",      "sack_positions"),
    "sack_boxes_dict":         ("sacks",      "sack_boxes_dict"),
    "sack_owner":              ("sacks",      "sack_owner"),
    "sack_scores":             ("sacks",      "sack_scores"),
    "sack_embeddings":         ("sacks",      "sack_embeddings"),
    # ground
    "sack_still_frames":       ("ground",     "sack_still_frames"),
    "ground_sacks":            ("ground",     "ground_sacks"),
    "ground_boxes":            ("ground",     "ground_boxes"),
    # delivery
    "person_sack_delivered":   ("delivery",   "person_sack_delivered"),
    "person_peak_delivery":    ("delivery",   "person_peak_delivery"),
    "total_sacks_counted":     ("delivery",   "total_sacks_counted"),
    "delivery_log":            ("delivery",   "delivery_log"),
    "anomaly_log":             ("delivery",   "anomaly_log"),
    "n_anomalies":             ("delivery",   "n_anomalies"),
    # peak
    "person_peak_count":       ("peak",       "person_peak_count"),
    "person_peak_used":        ("peak",       "person_peak_used"),
    "person_approach_fn":      ("peak",       "person_approach_fn"),
    "person_load":             ("peak",       "person_load"),
    # person position history
    "person_position_history":    ("persons",    "person_position_history"),
    # door zone (v21)
    "door_candidates":            ("door_state", "door_candidates"),
    "persons_past_door":          ("door_state", "persons_past_door"),
    # carrier guards (formerly "guards" / AssignBugState)
    "just_evicted_sacks":      ("guards",     "just_evicted_sacks"),
    "delivered_sacks":         ("guards", "delivered_sacks"),
    "sack_carrier_stamp":      ("guards",     "sack_carrier_stamp"),
    "orphaned_peaks":          ("guards",     "orphaned_peaks"),
    "orphaned_sack_owner":     ("guards",     "orphaned_sack_owner"),
}


class PipelineState:
    """
    Top-level container for all per-frame pipeline state.

    Provides a ``dict``-like interface (``[]``, ``get``, ``in``) so that
    existing call-sites using ``state["key"]`` work without modification
    while new code can access typed sub-groups directly via
    ``state.persons``, ``state.sacks``, etc.

    Direct attributes
    -----------------
    frame    : current BGR numpy frame (or None before first frame)
    persons  : PersonState
    sacks    : SackStateGroup
    ground   : GroundGroup
    delivery : DeliveryState
    peak     : PeakState
    guards   : CarrierGuardState
    """

    def __init__(self) -> None:
        self.frame:    Any               = None
        self._frame_no_val: int          = 0
        self.persons:  PersonState       = PersonState()
        self.sacks:    SackStateGroup    = SackStateGroup()
        self.ground:   GroundGroup       = GroundGroup()
        self.delivery: DeliveryState     = DeliveryState()
        self.peak:     PeakState         = PeakState()
        self.guards:    CarrierGuardState = CarrierGuardState()
        self.door_state: DoorState         = DoorState()

    # ── dict-like interface ───────────────────────────────────

    def __getitem__(self, key: str) -> Any:
        if key == "frame":
            return self.frame
        if key == "_frame_no":
            return self._frame_no_val
        group_attr, field_attr = _FIELD_MAP[key]
        return getattr(getattr(self, group_attr), field_attr)

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "frame":
            self.frame = value
            return
        if key == "_frame_no":
            self._frame_no_val = value
            return
        group_attr, field_attr = _FIELD_MAP[key]
        setattr(getattr(self, group_attr), field_attr, value)

    def __contains__(self, key: str) -> bool:
        return key in _FIELD_MAP or key in ("frame", "_frame_no")

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def pop(self, key: str, *args) -> Any:
        """Not supported on sub-group scalars; dict fields support .pop() directly."""
        raise TypeError(
            f"PipelineState.pop() is not supported for key {key!r}. "
            "Access the sub-group dict directly (e.g. state.persons.person_boxes.pop(...))."
        )
