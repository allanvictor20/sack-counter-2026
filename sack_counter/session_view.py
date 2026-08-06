"""
session_view.py — What a host sees of a running session.

Both counting loops accept a ``frame_sink`` callback, but they used to
hand it different things: entry mode passed its ``PipelineSession``,
exit mode passed its ``ExitPipelineState``.  Same parameter name, two
incompatible objects, so every consumer needed to know which mode it was
in and keep two code paths — which is exactly what the web console ended
up doing.

:class:`SessionView` is the one shape both modes produce.  It answers the
questions a host actually has — how many, by whom, what just happened —
without exposing the pipeline internals behind them.

Laziness
--------
The loop builds a view every frame; a host typically consumes ~15 a
second.  Everything derived is therefore a ``functools.cached_property``,
so constructing a view costs one object and consumers pay only for the
fields they read, once each.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Optional


@dataclass
class SessionView:
    """
    A read-only window onto the session as of one frame.

    Attributes:
        frame:     The annotated BGR frame.  This is the live buffer the
                   loop will draw over next — copy it if you keep it.
        frame_no:  Frame number, 1-based.
        fps:       Smoothed processing rate.
        mode:      ``"enter"`` or ``"exit"``.
        warning:   Operator-facing caveat, or None.
        session:   The PipelineSession (entry mode only).
        state:     The pipeline state container for this mode.
    """

    frame:    Any
    frame_no: int
    fps:      float
    mode:     str
    warning:  Optional[str] = None
    session:  Any = None
    state:    Any = None

    # ── Derived, computed at most once per view ───────────────

    @cached_property
    def label(self) -> str:
        """Headline label for the count — the design's kicker."""
        return "Sacks in" if self.mode == "enter" else "Sacks out"

    @cached_property
    def totals(self) -> dict:
        """Headline numbers, keyed the same way in both modes."""
        if self.mode == "exit":
            return {
                "sacks":   self.state["total_sacks_out"],
                "landed":  self.state["landing_exit_count"],
                "pile":    self.state["landing_peak_count"],
                "pending": len(self.state["tentative_crossings"]),
                "review":  len(self.state["discrepancy_flags"]),
                "label":   self.label,
            }
        return {
            "sacks":   self.state["total_sacks_counted"],
            "boxes":   self.session.box_pipeline.total_counted,
            "workers": len(self.workers),
            "review":  self.state["n_anomalies"],
            "relinks": len(self.session.relinker.reid_events),
            "label":   self.label,
        }

    @cached_property
    def workers(self) -> list[dict]:
        """
        Per-worker rows, richest first.

        Empty in exit mode, which counts sacks rather than carriers.
        Re-ID can map two raw ids onto one canonical worker, so rows are
        deduplicated on the canonical id.
        """
        if self.mode == "exit":
            return []
        state    = self.state
        relinker = self.session.relinker
        box_del  = self.session.box_pipeline.delivery_counts()

        rows, seen = [], set()
        for pid in state["confirmed_persons"]:
            can = relinker.canonical(pid)
            if can in seen:
                continue
            seen.add(can)
            rows.append({
                "id":        can,
                "delivered": len(state["person_sack_delivered"].get(can, set())),
                "boxes":     box_del.get(can, 0),
                "carrying":  state["person_load"].get(can, 0),
            })
        rows.sort(key=lambda w: (-w["delivered"], w["id"]))
        return rows

    @cached_property
    def events(self) -> list[dict]:
        """Newest-first activity feed, at most a dozen entries."""
        from .drawing import build_event_feed

        if self.mode == "exit":
            return [
                {"time": f"frame {rec.get('frame', 0)}",
                 "text": f"Sack {rec.get('sack_id')} left the room",
                 "tag": "Counted"}
                for rec in self.state["exit_log"][-12:][::-1]
            ]
        return build_event_feed(
            delivery_log     = self.state["delivery_log"],
            box_delivery_log = self.session.box_pipeline.delivery_log,
            anomaly_log      = self.state["anomaly_log"],
            reid_events      = self.session.relinker.reid_events,
            src_fps          = self.session.src_fps,
            limit            = 12,
        )

    # ── Constructors ──────────────────────────────────────────

    @classmethod
    def for_entry(cls, frame, session, frame_no: int, fps: float,
                  warning: str | None = None) -> "SessionView":
        """Build the view for an entry-mode frame."""
        return cls(frame=frame, frame_no=frame_no, fps=fps, mode="enter",
                   warning=warning, session=session, state=session.state)

    @classmethod
    def for_exit(cls, frame, state, frame_no: int, fps: float,
                 warning: str | None = None) -> "SessionView":
        """Build the view for an exit-mode frame."""
        return cls(frame=frame, frame_no=frame_no, fps=fps, mode="exit",
                   warning=warning, session=None, state=state)
