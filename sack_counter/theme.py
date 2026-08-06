"""
theme.py — Modernist design tokens and OpenCV drawing primitives.

The on-frame console is styled from the "Sack Counter Console" design
(design system ``modernist``).  Everything visual reads its colours,
type sizes and spacing from here so the overlay stays one coherent
surface instead of a pile of ad-hoc ``cv2.putText`` calls.

Nothing in this module touches pipeline state — it draws, measures and
nothing else.

Token translation
-----------------
The design system is authored in CSS hex; OpenCV wants BGR tuples.  The
constants below are the same values, byte-reversed.  Radii are all 0 in
the source system, which is why every panel here is a hard rectangle.

Scaling
-------
The design is laid out against a 1600 px-wide frame.  :func:`unit`
returns the multiplier for the frame actually being drawn on, so the
console keeps its proportions on 1280×720 footage and on 4K alike.
"""

from __future__ import annotations

import cv2
import numpy as np

# ── Palette (BGR) ──────────────────────────────────────────────
# --color-text / --color-bg / --color-surface
INK          = ( 29,  30,  32)   # #201e1d
PAPER        = (242, 242, 243)   # #f3f2f2
SURFACE      = (233, 233, 234)   # #eae9e9

# --color-accent ramp
ACCENT       = ( 19,  48, 236)   # #ec3013
ACCENT_200   = (217, 224, 255)   # #ffe0d9
ACCENT_300   = (184, 196, 255)   # #ffc4b8
ACCENT_400   = (131, 151, 255)   # #ff9783
ACCENT_500   = ( 60,  86, 255)   # #ff563c
ACCENT_600   = ( 15,  43, 221)   # #dd2b0f
ACCENT_700   = (  0,  24, 174)   # #ae1800
ACCENT_900   = ( 14,  23,  77)   # #4d170e

# --color-neutral ramp
NEUTRAL_200  = (231, 231, 234)   # #eae7e7
NEUTRAL_300  = (211, 211, 215)   # #d7d3d3
NEUTRAL_400  = (182, 182, 186)   # #bab6b6
NEUTRAL_500  = (151, 151, 155)   # #9b9797
NEUTRAL_600  = (121, 121, 125)   # #7d7979
NEUTRAL_700  = ( 93,  93,  96)   # #605d5d
NEUTRAL_800  = ( 65,  65,  68)   # #444141
NEUTRAL_900  = ( 43,  43,  45)   # #2d2b2b

# Panel washes used over live video, matching the design's
# rgba(32,30,29,.82) scrims.
SCRIM        = INK
SCRIM_ALPHA  = 0.82

# Archivo is a grotesk with a very heavy 800 weight for headings.  The
# closest pair OpenCV ships is DUPLEX (smoother, reads as the display
# face) for headings/numerals and SIMPLEX for body copy.
FONT_HEAD = cv2.FONT_HERSHEY_DUPLEX
FONT_BODY = cv2.FONT_HERSHEY_SIMPLEX

# The design's reference viewport width.
_REF_W = 1600.0

# Hershey glyphs are ~28 px tall at scale 1.0, so CSS pixels convert at
# roughly this ratio.
_PX_PER_SCALE = 28.0


# ── Scaling ────────────────────────────────────────────────────

def unit(frame_or_width) -> float:
    """
    Layout multiplier for *frame_or_width* against the 1600 px design.

    Clamped so the console stays readable on small footage and does not
    swallow the picture on very large frames.
    """
    width = (frame_or_width.shape[1]
             if hasattr(frame_or_width, "shape") else float(frame_or_width))
    return max(0.55, min(width / _REF_W, 1.75))


def fs(px: float, u: float = 1.0) -> float:
    """CSS pixel type size → OpenCV font scale."""
    return max(0.28, px * u / _PX_PER_SCALE)


def px(value: float, u: float = 1.0) -> int:
    """CSS pixel length → device pixels, never below 1."""
    return max(1, int(round(value * u)))


# ── Surfaces ───────────────────────────────────────────────────

def fill(frame, x1, y1, x2, y2, color, alpha: float = 1.0) -> None:
    """
    Paint a rectangle, optionally translucent.

    Only the destination region is blended, so a small panel costs a
    small blend — unlike copying the whole frame per overlay.
    """
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    if alpha >= 0.999:
        roi[:] = color
        return
    wash = np.empty_like(roi)
    wash[:] = color
    cv2.addWeighted(wash, alpha, roi, 1.0 - alpha, 0, roi)


def fill_polygon(frame, np_points, color, alpha: float = 0.10) -> None:
    """
    Wash a polygon with a translucent colour.

    Blends inside the polygon's bounding box only, so the cost scales
    with the shape rather than with the frame.
    """
    if alpha <= 0.0:
        return
    h, w = frame.shape[:2]
    pts = np.asarray(np_points).reshape(-1, 2)
    x1, y1 = np.clip(pts.min(axis=0), [0, 0], [w, h])
    x2, y2 = np.clip(pts.max(axis=0) + 1, [0, 0], [w, h])
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts - [x1, y1]], 255)
    wash = np.empty_like(roi)
    wash[:] = color
    blended = cv2.addWeighted(wash, alpha, roi, 1.0 - alpha, 0)
    np.copyto(roi, blended, where=mask[:, :, None].astype(bool))


def outline(frame, x1, y1, x2, y2, color, thickness: int = 1) -> None:
    """Hard-cornered stroke — the system has no border radius."""
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                  color, int(thickness), cv2.LINE_AA)


def rule(frame, x1, y, x2, color=NEUTRAL_700, thickness: int = 1,
         alpha: float = 1.0) -> None:
    """Horizontal divider — the design's 1–2 px hairlines."""
    fill(frame, x1, y, x2, y + thickness, color, alpha)


def edge(frame, x, y1, y2, color=ACCENT, thickness: int = 3) -> None:
    """Left accent bar used on callout cards."""
    fill(frame, x, y1, x + thickness, y2, color, 1.0)


# ── Type ───────────────────────────────────────────────────────

def text_w(s: str, scale: float, weight: int = 1,
           font: int = FONT_BODY, tracking: float = 0.0) -> int:
    """Advance width of *s* including letter-spacing."""
    if not s:
        return 0
    base = cv2.getTextSize(s, font, scale, weight)[0][0]
    return int(base + tracking * max(len(s) - 1, 0))


def text_h(scale: float, weight: int = 1, font: int = FONT_BODY) -> int:
    """Cap height of a line at *scale*."""
    return cv2.getTextSize("H", font, scale, weight)[0][1]


def text(frame, s: str, x, y, scale: float, color,
         weight: int = 1, font: int = FONT_BODY,
         tracking: float = 0.0) -> int:
    """
    Draw *s* with its baseline at *y*, returning the end x.

    ``tracking`` is per-character letter-spacing in device pixels; the
    design uses it heavily on uppercase kickers (``letter-spacing:.12em``).
    """
    x = int(x)
    if not s:
        return x
    if tracking <= 0.0:
        cv2.putText(frame, s, (x, int(y)), font, scale, color,
                    weight, cv2.LINE_AA)
        return x + cv2.getTextSize(s, font, scale, weight)[0][0]
    advance = 0.0
    for ch in s:
        cv2.putText(frame, ch, (x + int(advance), int(y)), font, scale,
                    color, weight, cv2.LINE_AA)
        advance += cv2.getTextSize(ch, font, scale, weight)[0][0] + tracking
    return x + int(advance)


def kicker(frame, s: str, x, y, u: float, color,
           size: float = 10.5, alpha_color=None) -> int:
    """
    Small uppercase label — the design's section eyebrow.

    ``font-size:10.5px; letter-spacing:.12em; text-transform:uppercase``.
    """
    scale = fs(size, u)
    return text(frame, s.upper(), x, y, scale, alpha_color or color,
                1, FONT_BODY, tracking=max(1.0, size * u * 0.12))


def wrap(s: str, max_w: int, scale: float, weight: int = 1,
         font: int = FONT_BODY, max_lines: int = 3) -> list[str]:
    """Greedy word wrap to *max_w* device pixels."""
    words = str(s).split()
    if not words:
        return []
    lines, current = [], words[0]
    for word in words[1:]:
        probe = f"{current} {word}"
        if text_w(probe, scale, weight, font) <= max_w:
            current = probe
        else:
            lines.append(current)
            current = word
            if len(lines) == max_lines:
                break
    if len(lines) < max_lines:
        lines.append(current)
    if len(lines) == max_lines:
        # Ellipsise the tail rather than dropping it silently.
        last = lines[-1]
        while last and text_w(last + "...", scale, weight, font) > max_w:
            last = last[:-1]
        consumed = sum(len(line.split()) for line in lines)
        if consumed < len(words):
            lines[-1] = last.rstrip() + "..."
    return lines


# ── Components ─────────────────────────────────────────────────

def chip(frame, s: str, x, y, u: float, bg, ink,
         size: float = 10.5, pad_x: float = 6.0, pad_y: float = 2.0,
         font: int = FONT_BODY, tracking: float = 0.0,
         weight: int = 1, border=None) -> tuple[int, int]:
    """
    Filled label chip — ``.tag`` in the design system.

    Drawn from its top-left corner.  Returns ``(width, height)`` so the
    caller can lay chips out in a row.
    """
    scale = fs(size, u)
    pad_x_px, pad_y_px = px(pad_x, u), px(pad_y, u)
    tw = text_w(s, scale, weight, font, tracking)
    th = text_h(scale, weight, font)
    w = tw + pad_x_px * 2
    h = th + pad_y_px * 2 + px(2, u)
    if bg is not None:
        fill(frame, x, y, x + w, y + h, bg, 1.0)
    if border is not None:
        outline(frame, x, y, x + w - 1, y + h - 1, border, 1)
    text(frame, s, x + pad_x_px, y + pad_y_px + th, scale, ink,
         weight, font, tracking)
    return w, h


def meter(frame, x, y, w, u: float, fraction: float,
          track=NEUTRAL_300, bar=ACCENT, height: float = 4.0,
          track_alpha: float = 1.0) -> int:
    """Thin progress bar — the per-worker share meter."""
    h = px(height, u)
    fill(frame, x, y, x + w, y + h, track, track_alpha)
    filled = int(w * max(0.0, min(float(fraction), 1.0)))
    if filled > 0:
        fill(frame, x, y, x + filled, y + h, bar, 1.0)
    return h


def dashed_polygon(frame, points, color, thickness: int = 2,
                   dash: int = 10, gap: int = 8, closed: bool = True) -> None:
    """
    Dashed outline — the landing zone's ``stroke-dasharray:10 8``.

    OpenCV has no dash support, so each edge is walked in dash+gap steps.
    """
    pts = [(int(p[0]), int(p[1])) for p in points]
    if len(pts) < 2:
        return
    period = max(dash + gap, 2)
    edges = list(zip(pts, pts[1:] + ([pts[0]] if closed else [])))
    for (x1, y1), (x2, y2) in edges:
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < 1.0:
            continue
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        travelled = 0.0
        while travelled < length:
            end = min(travelled + dash, length)
            cv2.line(
                frame,
                (int(x1 + ux * travelled), int(y1 + uy * travelled)),
                (int(x1 + ux * end),       int(y1 + uy * end)),
                color, thickness, cv2.LINE_AA,
            )
            travelled += period


def tracked_box(frame, x1, y1, x2, y2, u: float, stroke, label: str = "",
                label_ink=PAPER, thickness: float = 3.0,
                size: float = 10.5) -> None:
    """
    A detection box in the design's language: a hard stroke with a solid
    label chip seated on its top-left corner (``transform:translateY(-100%)``).
    """
    outline(frame, x1, y1, x2, y2, stroke, px(thickness, u))
    if not label:
        return
    scale = fs(size, u)
    th = text_h(scale, 1, FONT_BODY)
    pad_y = px(2, u)
    chip_h = th + pad_y * 2 + px(2, u)
    top = int(y1) - chip_h
    if top < 0:                      # box hugs the frame top — sit inside
        top = int(y1)
    chip(frame, label, int(x1), top, u, stroke, label_ink,
         size=size, pad_x=6.0, pad_y=2.0)


def contrast_ink(bg) -> tuple:
    """Pick PAPER or INK for text sitting on *bg* (BGR)."""
    b, g, r = bg[0], bg[1], bg[2]
    luma = 0.114 * b + 0.587 * g + 0.299 * r
    return INK if luma > 150 else PAPER
