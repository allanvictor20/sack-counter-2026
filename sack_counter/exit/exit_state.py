"""
exit_state.py — Typed state container for the exit sack counter.

Deliberately minimal compared to the entry counter state.py.
No carrier stamping, no peak windows, no person ReID needed.

State groups
------------
ExitSackState   — per-sack tracking (positions, projections, confirmation)
ExitCountState  — counters, logs, discrepancy flags
ExitPipelineState — top-level container with dict-like interface
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..pipeline.field_view import FieldView


# ── Sub-group dataclasses ─────────────────────────────────────

@dataclass
class ExitSackState:
    """Per-sack detection and crossing state."""

    # sid -> (cx, cy) of sack centroid this frame
    sack_positions: dict = field(default_factory=dict)

    # sid -> (x1, y1, x2, y2) bounding box
    sack_boxes: dict = field(default_factory=dict)

    # sid -> detection confidence score
    sack_conf: dict = field(default_factory=dict)

    # sid -> projection value onto door normal from LAST frame
    sack_prev_proj: dict = field(default_factory=dict)

    # sid -> projection value onto door normal THIS frame
    sack_curr_proj: dict = field(default_factory=dict)

    # sids confirmed present this frame
    confirmed_sacks: set = field(default_factory=set)

    # hit/miss counters for confirmation hysteresis
    sack_hit: dict = field(default_factory=dict)   # sid -> int
    sack_miss: dict = field(default_factory=dict)  # sid -> int

    # sid -> frame number when TENTATIVE crossing was logged
    tentative_crossings: dict = field(default_factory=dict)

    # sids already counted as exited — never recount
    crossed_sacks: set = field(default_factory=set)

    # sids confirmed stationary inside landing zone
    landed_sacks: set = field(default_factory=set)

    # Subset of landed_sacks that are ALIASES — a second (or third) track
    # ID assigned to a sack that was already landed, recognised by spatial
    # dedup.  Held separately so the distinct physical sack count is
    # len(landed_sacks) - len(landed_aliases).
    landed_aliases: set = field(default_factory=set)

    # sids that have been picked up from landing zone (for logging)
    carried_away_sacks: set = field(default_factory=set)


@dataclass
class ExitCountState:
    """Exit event counters and logs."""

    # Primary exit counter — incremented at CROSSING confirmation
    total_sacks_out: int = 0

    # Highest number of distinct sacks ever seen simultaneously in landing zone
    # Only goes up — used to catch sacks missed at the door crossing
    landing_peak_count: int = 0

    # How many sacks the landing zone thinks have exited
    # (compared against total_sacks_out for discrepancy detection)
    landing_exit_count: int = 0

    # One entry per confirmed exit event
    exit_log: list = field(default_factory=list)

    # Entries where door_count and landing_count disagree
    discrepancy_flags: list = field(default_factory=list)

    # Current frame number (updated every frame)
    frame_no: int = 0

    # sid -> (cx, cy) of every landed sack's position, used for spatial
    # deduplication when a new track ID appears at the same physical
    # location (see _find_nearby_landed_sack in exit_landing.py).
    landed_positions: dict = field(default_factory=dict)

    # sid -> frame number this landed sack's position was last refreshed.
    # Used alongside landed_positions for TIME-AWARE spatial dedup: a
    # new sack ID near an old landed position only counts as "same
    # physical sack, new ID" if that old position was active RECENTLY
    # (within a short window), not just "anywhere near any sack ever
    # landed in the whole session". This prevents two genuinely separate
    # sacks landing close together at different times from being
    # incorrectly merged into one.
    landed_last_seen_frame: dict = field(default_factory=dict)


# ── Flat-key map: logical name → (group_attr, field_attr) ────

_FIELD_MAP: dict[str, tuple[str, str]] = {
    # sack state
    "sack_positions":       ("sacks", "sack_positions"),
    "sack_boxes":           ("sacks", "sack_boxes"),
    "sack_conf":            ("sacks", "sack_conf"),
    "sack_prev_proj":       ("sacks", "sack_prev_proj"),
    "sack_curr_proj":       ("sacks", "sack_curr_proj"),
    "confirmed_sacks":      ("sacks", "confirmed_sacks"),
    "sack_hit":             ("sacks", "sack_hit"),
    "sack_miss":            ("sacks", "sack_miss"),
    "tentative_crossings":  ("sacks", "tentative_crossings"),
    "crossed_sacks":        ("sacks", "crossed_sacks"),
    "landed_sacks":         ("sacks", "landed_sacks"),
    "landed_aliases":       ("sacks", "landed_aliases"),
    "carried_away_sacks":   ("sacks", "carried_away_sacks"),
    # count state
    "total_sacks_out":      ("counts", "total_sacks_out"),
    "landing_peak_count":   ("counts", "landing_peak_count"),
    "landing_exit_count":   ("counts", "landing_exit_count"),
    "exit_log":             ("counts", "exit_log"),
    "discrepancy_flags":    ("counts", "discrepancy_flags"),
    "frame_no":             ("counts", "frame_no"),
    "landed_positions":     ("counts", "landed_positions"),
    "landed_last_seen_frame": ("counts", "landed_last_seen_frame"),
}


@dataclass
class ExitPipelineState(FieldView):
    """
    Top-level state container.

    Exposes a dict-like interface (``state["key"]``) so callers do not
    need to know about the sub-group structure.  The interface is
    :class:`FieldView`, shared with the entry-mode container so the two
    modes cannot drift apart again.

    This used to keep an ``_extra`` overflow dict for unmapped keys,
    which meant a misspelled field silently became a new one and the
    correct name kept reading its default forever.  Unknown keys now
    raise, at the assignment that made the mistake.
    """

    sacks:  ExitSackState  = field(default_factory=ExitSackState)
    counts: ExitCountState = field(default_factory=ExitCountState)

    _FIELDS = _FIELD_MAP

    def reset(self) -> None:
        """Reset all state to initial values (for unit tests)."""
        self.sacks  = ExitSackState()
        self.counts = ExitCountState()
