"""
exit_drawing.py — Frame annotation for the exit sack counter.

Same console as entry mode, counting the other direction.  The chrome
(headline block, scrim panels, status footer) is shared with
``drawing.py`` so a session looks the same whichever way the sacks are
going; only the numbers and the geometry differ.

Draws:
  - Door polygon (accent, reusing DoorPolygon.draw)
  - Landing zone polygon (dashed paper outline, faint wash)
  - Per-sack bounding boxes with state colour coding:
      Neutral 300  = detected, not confirmed
      Paper        = confirmed, no event
      Accent 400   = TENTATIVE crossing
      Accent       = CONFIRMED exit (crossed_sacks)
      Accent 500   = LANDED (stationary in landing zone)
  - Headline count block, session panel and status footer
  - "Check this" callout if the two counts disagree
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from .. import theme as T
from ..drawing import (
    draw_count_block, draw_stat_panel, draw_status_strip, draw_warning_card,
)
from .exit_landing import reconcile_counts

if TYPE_CHECKING:
    from .exit_state import ExitPipelineState
    from ..door_polygon import DoorPolygon
    from .landing_zone import LandingZone


# Sack state colours, mapped onto the design's accent ramp.  Only the
# states that mean "this one counted" get the full accent.
_COL_DETECTED   = T.NEUTRAL_300   # seen but not confirmed
_COL_CONFIRMED  = T.PAPER         # confirmed, no event
_COL_TENTATIVE  = T.ACCENT_400    # tentative crossing
_COL_EXITED     = T.ACCENT        # confirmed exit
_COL_LANDED     = T.ACCENT_500    # stationary in landing zone


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
    u = T.unit(frame)

    # ── Geometry ──────────────────────────────────────────────
    best_count = reconcile_counts(state)
    door.draw(
        frame,
        label="EXIT DOOR",
        count=best_count,
        show_debug=True,
    )
    landing_zone.draw(
        frame,
        label="LANDING ZONE",
        count=state["landing_peak_count"],
    )

    # ── Sack bounding boxes ───────────────────────────────────
    # Unconfirmed detections stay quiet — a hairline, no label.
    if all_detections:
        for (sid, x1, y1, x2, y2, cx, cy, conf) in all_detections:
            if sid not in state["confirmed_sacks"]:
                T.outline(frame, x1, y1, x2, y2, _COL_DETECTED, T.px(1.5, u))

    # Confirmed sacks — colour by state
    for sid in state["confirmed_sacks"]:
        box = state["sack_boxes"].get(sid)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        conf = state["sack_conf"].get(sid, 0.0)

        if sid in state["crossed_sacks"]:
            colour = _COL_EXITED
            label  = f"Sack {sid} · out"
        elif sid in state["landed_sacks"]:
            colour = _COL_LANDED
            label  = f"Sack {sid} · landed"
        elif sid in state["tentative_crossings"]:
            colour = _COL_TENTATIVE
            label  = f"Sack {sid} · crossing"
        else:
            colour = _COL_CONFIRMED
            label  = f"Sack {sid} · {conf:.2f}"

        T.tracked_box(frame, x1, y1, x2, y2, u, colour, label,
                      label_ink=T.contrast_ink(colour),
                      thickness=2.5, size=10.5)

    # ── Console chrome ────────────────────────────────────────
    _draw_console(frame, u, state, fps, fn, best_count)


def _draw_console(frame, u, state, fps: float, fn: int, best_count: int) -> None:
    """Headline block, session panel, discrepancy callout, status footer.

    The headline "Sacks out" number is the reconciled count — driven
    primarily by the landing zone (landing_exit_count), since that is
    the module's primary, more reliable counter (see exit_landing.py
    module docstring). Door-crossing only raises this number further
    if --use-door-crossing ever counts higher than the landing zone
    (e.g. a sack landed somewhere the landing zone polygon doesn't
    cover); it never lowers it, and by default (door-crossing off)
    total_sacks_out is always 0, so best_count == landing_exit_count.
    """
    fw = frame.shape[1]

    draw_count_block(
        frame, u, "Sacks out", best_count,
        stats=(("On the floor", state["landing_peak_count"]),
               ("Pending", len(state["tentative_crossings"]))),
    )

    panel_w = int(min(max(fw * 0.20, T.px(190, u)), T.px(290, u)))
    panel_x = fw - panel_w
    cursor  = 0

    # The two counts disagreeing is the one thing an operator must see.
    if state["discrepancy_flags"]:
        flag = state["discrepancy_flags"][-1]
        cursor += draw_warning_card(
            frame, u, panel_x, cursor, panel_w,
            f"The door counted {flag['door_count']} sacks and the landing "
            f"zone counted {flag['landing_count']}. Check the video around "
            f"this point before trusting the total.",
        ) + T.px(2, u)

    draw_stat_panel(
        frame, u, panel_x, cursor, panel_w, "This session",
        rows=(
            ("Sacks out (best est.)", best_count, T.ACCENT_500),
            ("Landed in the zone",    state["landing_exit_count"]),
            ("On the floor now",      state["landing_peak_count"]),
            ("Counted at the door",   state["total_sacks_out"]),
            ("Waiting to confirm",    len(state["tentative_crossings"])),
        ),
    )

    draw_status_strip(
        frame, u, "Counting sacks out", fn, fps, None,
        f"{len(state['confirmed_sacks'])} tracked  ·  "
        f"{len(state['landed_sacks'])} landed",
    )
