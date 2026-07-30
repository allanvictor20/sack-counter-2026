"""
exit_drawing.py — Frame annotation for the exit sack counter.

Draws:
  - Door polygon (cyan, reusing DoorPolygon.draw)
  - Landing zone polygon (green, semi-transparent fill)
  - Per-sack bounding boxes with state colour coding:
      Grey    = detected, not confirmed
      White   = confirmed, not near door
      Yellow  = TENTATIVE crossing
      Green   = CONFIRMED exit (crossed_sacks)
      Teal    = LANDED (stationary in landing zone)
  - HUD overlay: total exits, landing count, FPS, frame number
  - Discrepancy warning banner if counts diverge
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .exit_state import ExitPipelineState
    from ..door_polygon import DoorPolygon
    from .landing_zone import LandingZone


# Colour palette (BGR)
_COL_DETECTED   = (160, 160, 160)   # grey   — seen but not confirmed
_COL_CONFIRMED  = (220, 220, 220)   # white  — confirmed, no event
_COL_TENTATIVE  = (0,   200, 255)   # yellow — tentative crossing
_COL_EXITED     = (0,   255, 0)     # green  — confirmed exit
_COL_LANDED     = (0,   220, 180)   # teal   — stationary in landing zone
_COL_HUD_BG     = (20,  20,  20)    # near-black HUD background
_COL_HUD_TEXT   = (255, 255, 255)   # white HUD text
_COL_WARN       = (0,   0,   220)   # red    — discrepancy warning


def draw_exit_frame(
    frame: np.ndarray,
    state: "ExitPipelineState",
    door: "DoorPolygon",
    landing_zone: "LandingZone",
    fps: float,
    fn: int,
    all_detections: list[tuple] | None = None,
) -> None:
    """
    Draw all exit-mode overlays onto *frame* in-place.

    Args:
        frame:          BGR numpy array (modified in-place).
        state:          ExitPipelineState.
        door:           Calibrated DoorPolygon.
        landing_zone:   Calibrated LandingZone.
        fps:            Current smoothed FPS.
        fn:             Current frame number.
        all_detections: Raw (sid, x1, y1, x2, y2, cx, cy, conf) list
                        (optional, for drawing unconfirmed detections).
    """
    # ── Door polygon ──────────────────────────────────────────
    door.draw(
        frame,
        color=(0, 200, 255),
        label="EXIT DOOR",
        count=state["total_sacks_out"],
        show_debug=True,
    )

    # ── Landing zone ──────────────────────────────────────────
    from .landing_zone import LandingZone as _LZ
    landing_zone.draw(
        frame,
        color=(0, 255, 180),
        label="LANDING",
        count=state["landing_peak_count"],
        fill_alpha=0.15,
    )

    # ── Sack bounding boxes ───────────────────────────────────
    # Unconfirmed detections (grey, thin)
    if all_detections:
        for (sid, x1, y1, x2, y2, cx, cy, conf) in all_detections:
            if sid not in state["confirmed_sacks"]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), _COL_DETECTED, 1)

    # Confirmed sacks — colour by state
    for sid in state["confirmed_sacks"]:
        box = state["sack_boxes"].get(sid)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        conf = state["sack_conf"].get(sid, 0.0)

        if sid in state["crossed_sacks"]:
            colour = _COL_EXITED
            label  = f"OUT#{sid}"
        elif sid in state["landed_sacks"]:
            colour = _COL_LANDED
            label  = f"LAND#{sid}"
        elif sid in state["tentative_crossings"]:
            colour = _COL_TENTATIVE
            label  = f"?#{sid}"
        else:
            colour = _COL_CONFIRMED
            label  = f"S#{sid} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            frame, label,
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1,
        )

    # ── HUD ───────────────────────────────────────────────────
    _draw_hud(frame, state, fps, fn)

    # ── Discrepancy banner ────────────────────────────────────
    if state["discrepancy_flags"]:
        last_flag = state["discrepancy_flags"][-1]
        diff = last_flag.get("diff", 0)
        banner = (
            f"! COUNT DISCREPANCY: door={last_flag['door_count']} "
            f"landing={last_flag['landing_count']} diff={diff}"
        )
        fh = frame.shape[0]
        cv2.rectangle(frame, (0, fh - 36), (frame.shape[1], fh),
                      _COL_WARN, -1)
        cv2.putText(
            frame, banner,
            (8, fh - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1,
        )


def _draw_hud(
    frame: np.ndarray,
    state: "ExitPipelineState",
    fps: float,
    fn: int,
) -> None:
    """Draw the top-right HUD panel."""
    fw = frame.shape[1]

    lines = [
        f"MODE: EXIT COUNTER",
        f"SACKS OUT  : {state['total_sacks_out']}",
        f"LANDING CT : {state['landing_exit_count']}",
        f"PILE NOW   : {state['landing_peak_count']}",
        f"TENTATIVE  : {len(state['tentative_crossings'])}",
        f"FPS        : {fps:.1f}",
        f"FRAME      : {fn}",
    ]

    line_h  = 22
    padding = 6
    panel_w = 210
    panel_h = len(lines) * line_h + padding * 2

    x0 = fw - panel_w - 8
    y0 = 8

    # Background
    cv2.rectangle(frame,
                  (x0, y0),
                  (x0 + panel_w, y0 + panel_h),
                  _COL_HUD_BG, -1)
    cv2.rectangle(frame,
                  (x0, y0),
                  (x0 + panel_w, y0 + panel_h),
                  (80, 80, 80), 1)

    for i, line in enumerate(lines):
        ty = y0 + padding + (i + 1) * line_h - 4
        # Highlight the main counter in green
        colour = (0, 255, 120) if i == 1 else _COL_HUD_TEXT
        cv2.putText(
            frame, line,
            (x0 + padding, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, colour, 1,
        )
