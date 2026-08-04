"""
landing_zone.py — Landing zone polygon for the exit sack counter.

The landing zone is a user-drawn polygon on the corridor floor directly
outside the door.  Any sack that appears stationary inside this zone has
definitively exited the room.

The LandingZone class mirrors the DoorPolygon API where useful (contains,
draw) so the two can be used symmetrically in exit_main.py.

Calibration
-----------
calibrate_landing_zone() is called immediately after calibrate_door_polygon()
during exit-mode startup.  It shows the same first frame with the door polygon
already drawn so the user can see both zones simultaneously.
"""

from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional

from .. import theme as T
from ..door_polygon import _get_display_cap


@dataclass
class LandingZone:
    """
    Arbitrary polygon (3–8 points) marking the corridor floor area
    where thrown sacks land.

    Args:
        points: List of (x, y) tuples (3 to 8 points).

    Attributes
    ----------
    points      : validated list of (x, y) tuples
    np_points   : numpy array shape (N, 1, 2) dtype int32
    centroid    : (cx, cy) centroid of the polygon
    bbox        : (x1, y1, x2, y2) bounding box
    """

    points: list  # list of (x, y) tuples

    def __post_init__(self):
        if len(self.points) < 3:
            raise ValueError(
                f"LandingZone requires at least 3 points, got {len(self.points)}."
            )
        if len(self.points) > 8:
            raise ValueError(
                f"LandingZone accepts at most 8 points, got {len(self.points)}."
            )

        pts = np.array(self.points, dtype=np.int32)
        self.np_points = pts.reshape(-1, 1, 2)

        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        self.centroid = (int(np.mean(xs)), int(np.mean(ys)))
        self.bbox = (min(xs), min(ys), max(xs), max(ys))

    # ── Geometry ───────────────────────────────────────────────

    def contains(self, x: int, y: int) -> bool:
        """Return True if (x, y) is inside the landing zone polygon."""
        result = cv2.pointPolygonTest(self.np_points, (float(x), float(y)), False)
        return result >= 0

    def bbox_overlaps(self, x1: int, y1: int, x2: int, y2: int,
                      min_fraction: float = 0.30) -> bool:
        """
        Return True if at least *min_fraction* of the bounding box's area
        falls inside the landing zone polygon.

        Centroid-only checking (the previous approach) misses sacks that
        are mostly inside the zone but happen to straddle the boundary —
        very common at the edge of a pile, where a sack's centroid can
        sit just outside the polygon even though most of the sack visibly
        sits inside it.  This caused real sacks at the zone edge to be
        silently ignored.

        This implementation samples the bounding box on a small grid and
        counts what fraction of sample points fall inside the polygon —
        cheap to compute and accurate enough for sack-sized boxes (no need
        for exact polygon-clipping geometry).

        Args:
            x1, y1, x2, y2: Bounding box corners.
            min_fraction:   Minimum fraction of the box that must be
                            inside the zone to count as "in the zone".
                            Default 0.30 — roughly a third of the sack
                            must be visibly inside.

        Returns:
            True if the box sufficiently overlaps the landing zone.
        """
        # Fast path: centroid inside -> definitely overlapping enough
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        if self.contains(cx, cy):
            return True

        # Fast rejection: bounding boxes don't even intersect at all
        zx1, zy1, zx2, zy2 = self.bbox
        if x2 < zx1 or x1 > zx2 or y2 < zy1 or y1 > zy2:
            return False

        # Sample a grid of points across the sack's bounding box and
        # count what fraction land inside the landing zone polygon.
        grid_n = 5  # 5x5 = 25 sample points; cheap and accurate enough
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        inside_count = 0
        total_count = 0
        for gy in range(grid_n):
            for gx in range(grid_n):
                px = x1 + int(w * (gx + 0.5) / grid_n)
                py = y1 + int(h * (gy + 0.5) / grid_n)
                total_count += 1
                if self.contains(px, py):
                    inside_count += 1

        if total_count == 0:
            return False

        fraction_inside = inside_count / total_count
        return fraction_inside >= min_fraction

    # ── Drawing ────────────────────────────────────────────────

    def draw(
        self,
        frame,
        color: tuple | None = None,
        thickness: int | None = None,
        label: str = "LANDING ZONE",
        count: int = 0,
        fill_alpha: float = 0.06,
    ) -> None:
        """
        Draw the landing zone polygon with a semi-transparent fill.

        Styled as the design's secondary geometry: a dashed paper outline
        over a 6% wash, so it never competes with the accent-red door.

        Args:
            frame:       BGR numpy array (modified in-place).
            color:       Polygon edge and label colour (B, G, R).  Defaults
                         to the design's paper.
            thickness:   Edge line thickness.  Defaults to 2 px, scaled.
            label:       Text shown at centroid.
            count:       Current sack count shown next to label.
            fill_alpha:  Opacity of the polygon fill (0 = invisible).
        """
        u = T.unit(frame)
        if color is None:
            color = T.PAPER
        if thickness is None:
            thickness = T.px(2, u)

        T.fill_polygon(frame, self.np_points, color, alpha=fill_alpha)
        T.dashed_polygon(frame, self.points, color, thickness,
                         dash=T.px(10, u), gap=T.px(8, u))

        # Corner markers
        marker = T.px(7, u)
        for px_, py_ in self.points:
            T.fill(frame, px_ - marker // 2, py_ - marker // 2,
                   px_ - marker // 2 + marker, py_ - marker // 2 + marker,
                   color, 1.0)

        # Label — uppercase kicker over the count, same rhythm as the door,
        # anchored to the zone's lowest corner so it clears the sacks in it.
        kick_scale = T.fs(11, u)
        num_scale  = T.fs(20, u)
        kick_h = T.text_h(kick_scale, 1, T.FONT_BODY)
        num_h  = T.text_h(num_scale, 2, T.FONT_HEAD)
        anchor = max(self.points, key=lambda p: p[1])
        lx = anchor[0] + T.px(14, u)
        base = anchor[1] + T.px(10, u) + kick_h
        if base + T.px(6, u) + num_h > frame.shape[0]:
            base = anchor[1] - T.px(10, u) - num_h - T.px(6, u)
        T.text(frame, label.upper(), lx, base, kick_scale, color, 1,
               T.FONT_BODY, tracking=max(1.0, 11 * u * 0.12))
        T.text(frame, str(count), lx, base + T.px(6, u) + num_h,
               num_scale, color, 2, T.FONT_HEAD)


# ── Calibration ────────────────────────────────────────────────

def _draw_zone_calibration_overlay(display, points, error_msg) -> None:
    """
    Render the landing-zone calibration overlay in place.

    Presentation only — the click list is read, never written.  Matches
    the door calibration screen so the two steps feel like one flow.
    """
    u = T.unit(display)
    fh, fw = display.shape[:2]

    if len(points) >= 3:
        T.fill_polygon(display, np.array(points, dtype=np.int32),
                       T.PAPER, alpha=0.06)
        T.dashed_polygon(display, points, T.PAPER, T.px(2, u),
                         dash=T.px(10, u), gap=T.px(8, u))
    else:
        for i in range(len(points) - 1):
            cv2.line(display, points[i], points[i + 1], T.PAPER,
                     T.px(2, u), cv2.LINE_AA)

    marker = T.px(20, u)
    mark_scale = T.fs(13, u)
    for i, (px_, py_) in enumerate(points):
        T.fill(display, px_ - marker // 2, py_ - marker // 2,
               px_ - marker // 2 + marker, py_ - marker // 2 + marker,
               T.PAPER, 1.0)
        digit = str(i + 1)
        dw = T.text_w(digit, mark_scale, 1, T.FONT_HEAD)
        dh = T.text_h(mark_scale, 1, T.FONT_HEAD)
        T.text(display, digit, px_ - dw // 2, py_ + dh // 2,
               mark_scale, T.INK, 1, T.FONT_HEAD)

    # Header card, top-left
    pad_x, pad_y = T.px(16, u), T.px(12, u)
    title_scale = T.fs(15, u)
    body_scale  = T.fs(12.5, u)
    title_h = T.text_h(title_scale, 1, T.FONT_HEAD)
    body_h  = T.text_h(body_scale, 1, T.FONT_BODY)
    body = ("Click three or more corners around the floor where sacks land. "
            f"{len(points)} marked — Enter to save.")
    lines = T.wrap(body, T.px(330, u) - pad_x * 2, body_scale, 1,
                   T.FONT_BODY, max_lines=3)
    card_h = pad_y * 2 + title_h + T.px(6, u) + int(body_h * 1.5) * len(lines)
    T.fill(display, 0, 0, T.px(330, u), card_h, T.SCRIM, T.SCRIM_ALPHA)
    T.text(display, "Mark the landing zone", pad_x, pad_y + title_h,
           title_scale, T.PAPER, 1, T.FONT_HEAD)
    by = pad_y + title_h + T.px(6, u) + body_h
    for line in lines:
        T.text(display, line, pad_x, by, body_scale, T.NEUTRAL_400,
               1, T.FONT_BODY)
        by += int(body_h * 1.5)

    if error_msg:
        err_scale = T.fs(12.5, u)
        err_h = T.text_h(err_scale, 1, T.FONT_BODY)
        bar_h = pad_y * 2 + err_h
        bar_w = min(fw, pad_x * 2 + T.px(3, u)
                    + T.text_w(error_msg, err_scale, 1, T.FONT_BODY))
        T.fill(display, 0, fh - bar_h, bar_w, fh, T.ACCENT_200, 0.94)
        T.edge(display, 0, fh - bar_h, fh, T.ACCENT, T.px(3, u))
        T.text(display, error_msg, T.px(3, u) + pad_x, fh - pad_y,
               err_scale, T.ACCENT_900, 1, T.FONT_BODY)



def calibrate_landing_zone(
    cap,
    door_polygon=None,
    cfg: dict = None,
) -> "LandingZone":
    """
    Interactive calibration: show first frame, let user click 3–8 points
    to define the landing zone on the corridor floor.

    The door polygon (if provided) is drawn on the frame so the user can
    see both zones simultaneously.

    Controls:
        Left-click  : Add a point
        Right-click : Remove last point
        Enter/Space : Confirm (requires ≥ 3 points)
        r           : Reset all points
        q           : Quit (raises RuntimeError)

    Args:
        cap:          cv2.VideoCapture — seeks to frame 0.
        door_polygon: Optional DoorPolygon to draw for reference.
        cfg:          Optional config dict updated with landing_zone_points.

    Returns:
        LandingZone with the user-selected points.

    Raises:
        RuntimeError: if the user quits without completing calibration.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame for landing zone calibration.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── Fit large frames to a reasonable on-screen window size ─────
    # Same fix as door_polygon.calibrate_door_polygon — see that function
    # for the full explanation. Display is scaled down; clicks are
    # converted back to original-frame coordinates via disp_scale.
    max_w, max_h = _get_display_cap()
    fh, fw = frame.shape[:2]
    disp_scale = min(max_w / fw, max_h / fh, 1.0)  # never upscale
    disp_w, disp_h = int(fw * disp_scale), int(fh * disp_scale)

    points: list = []
    error_msg: str = ""
    clone = frame.copy()

    # Draw existing door polygon for reference
    if door_polygon is not None:
        door_polygon.draw(clone, label="DOOR", count=0, show_debug=False)

    WIN = "Calibrate Landing Zone | Enter=confirm | r=reset | q=quit"

    def _mouse(event, x, y, flags, param):
        ox, oy = int(x / disp_scale), int(y / disp_scale)
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 8:
                points.append((ox, oy))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if points:
                points.pop()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, disp_w, disp_h)
    cv2.moveWindow(WIN, 0, 0)
    cv2.setMouseCallback(WIN, _mouse)

    while True:
        display = clone.copy()

        _draw_zone_calibration_overlay(display, points, error_msg)

        cv2.imshow(WIN, cv2.resize(display, (disp_w, disp_h)))
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32):  # Enter or Space
            if len(points) >= 3:
                error_msg = ""
                break
            else:
                error_msg = f"Need at least 3 points (have {len(points)})"

        elif key == ord("r"):
            points.clear()
            error_msg = ""

        elif key == ord("q"):
            cv2.destroyWindow(WIN)
            raise RuntimeError("Landing zone calibration cancelled by user.")

    cv2.destroyWindow(WIN)

    zone = LandingZone(points=points)

    if cfg is not None:
        cfg["landing_zone_points"] = zone.points

    return zone
