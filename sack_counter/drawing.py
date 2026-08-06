"""
drawing.py — OpenCV drawing for the Sack Counter console overlay.

The overlay is a rendering of the "Sack Counter Console" design: one
accent block carrying the number that matters, a scrim column on the
right for who is counting and what just happened, and a quiet mono
footer for the machine's own vitals.  All colour, type and spacing come
from :mod:`sack_counter.theme` — nothing here invents a value.

Public functions (signatures unchanged):
    draw_person_box
    draw_box_detections
    draw_sack_boxes
    draw_hud

Shared console chrome, also used by the exit-mode overlay:
    draw_count_block
    draw_warning_card
    draw_stat_panel
    draw_status_strip

:func:`build_event_feed` is a pure read-only projection of the pipeline's
existing logs into the design's "What just happened" list.  It records
nothing and mutates nothing.
"""

import cv2
from datetime import datetime

from . import theme as T
from .colors import (
    C_PERSON, C_SACK, C_SACK_ST, C_OWNED,
    C_GHOST, C_COUNTED, C_ANOMALY, C_REID, C_BOX,
    C_LOW_CONF, C_MED_CONF,
)
from .version import VERSION_TAG as _VERSION_TAG
from .confidence import ConfidenceTracker, confidence_class
from .trackers import SackMotionTracker, GhostSacks

# Opacity of the extrapolated-ghost overlay.  Deliberately faint: ghosts
# are a prediction, not a detection, and must be visually distinguishable
# from a real box.
_GHOST_ALPHA = 0.45

# Tag styles from the design system's .tag variants, as (bg, ink, border).
_TAG_ACCENT  = (T.ACCENT_200, T.ACCENT_900, None)
_TAG_NEUTRAL = (T.NEUTRAL_200, T.NEUTRAL_800, None)
_TAG_OUTLINE = (None, T.ACCENT_500, T.ACCENT_500)


def draw_person_box(frame, box, pid, canonical_pid,
                    current_sacks=0, delivered_sacks=0,
                    current_boxes=0, delivered_boxes=0,
                    counted=False, was_relinked=False,
                    cs=None, ds=None, cb=None, db=None,
                    delivered=None, relinked=None):
    # Accept both old positional names and new short kwargs
    if cs is not None:       current_sacks   = cs
    if ds is not None:       delivered_sacks = ds
    if cb is not None:       current_boxes   = cb
    if db is not None:       delivered_boxes = db
    if delivered is not None: counted        = delivered
    if relinked is not None:  was_relinked   = relinked

    x1, y1, x2, y2 = box
    u = T.unit(frame)

    # Accent means "this one is carrying / has carried"; paper means
    # "tracked, carrying nothing" — exactly the two track colours in the
    # design's live view.
    carrying = current_sacks > 0 or current_boxes > 0
    if was_relinked:
        stroke = C_REID
    elif counted or carrying:
        stroke = C_COUNTED
    else:
        stroke = C_PERSON

    # Label chip: "Worker 3 · 2 sacks"
    label = f"Worker {canonical_pid}"
    if pid != canonical_pid:
        label += f" ({pid})"
    if current_sacks:
        label += f" · {current_sacks} sack" + ("s" if current_sacks != 1 else "")
    if current_boxes:
        label += f" · {current_boxes} box" + ("es" if current_boxes != 1 else "")

    T.tracked_box(frame, x1, y1, x2, y2, u, stroke, label,
                  label_ink=T.contrast_ink(stroke),
                  thickness=3.0, size=10.5)

    # Delivered totals ride a scrim strip on the bottom edge — the number
    # the operator actually reports at the end of the shift.
    kick_scale = T.fs(9.5, u)
    num_scale  = T.fs(17, u)
    kick_h     = T.text_h(kick_scale, 1, T.FONT_BODY)
    num_h      = T.text_h(num_scale, 2, T.FONT_HEAD)
    pad        = T.px(6, u)
    strip_h    = pad * 2 + max(num_h, kick_h)
    top        = max(int(y1), int(y2) - strip_h)
    T.fill(frame, x1, top, x2, y2, T.SCRIM, T.SCRIM_ALPHA)

    baseline = y2 - pad
    cursor   = int(x1) + pad
    for kick, value, ink in (("Sacks", delivered_sacks, T.ACCENT_500),
                             ("Boxes", delivered_boxes, T.PAPER)):
        cursor = T.kicker(frame, kick, cursor, baseline, u,
                          T.NEUTRAL_400, size=9.5) + T.px(5, u)
        cursor = T.text(frame, str(value), cursor, baseline, num_scale,
                        ink, 2, T.FONT_HEAD) + T.px(12, u)
        if cursor > x2 - pad:
            break


def draw_box_detections(frame, raw_boxes, box_owner,
                        conf_tracker: ConfidenceTracker, cfg: dict,
                        ground_boxes: set = None):
    if ground_boxes is None:
        ground_boxes = set()
    u = T.unit(frame)
    for (bid, x1, y1, x2, y2, sc) in raw_boxes:
        if bid in ground_boxes:                          # unowned still boxes
            continue
        pid       = box_owner.get(bid)
        conf      = conf_tracker.delivery_confidence(bid)
        own_score = conf_tracker._own.get(bid, 0.0)
        if conf >= conf_tracker.min:
            if own_score >= cfg["conf_class_high"]:
                color = C_BOX
            elif own_score >= cfg["conf_class_medium"]:
                color = C_MED_CONF
            else:
                color = C_LOW_CONF
        else:
            color = C_LOW_CONF
        # Secondary detections stay quiet: hairline stroke, no filled chip.
        T.outline(frame, x1, y1, x2, y2, color, T.px(1.5, u))
        conf_lbl = confidence_class(own_score, cfg)
        lbl = f"Box {bid}" + (f" → W{pid} · {conf_lbl.lower()}" if pid else "")
        T.text(frame, lbl, x1 + T.px(2, u), y1 - T.px(4, u),
               T.fs(9.5, u), color, 1, T.FONT_BODY)


def draw_sack_boxes(frame, raw_sacks, sack_owner, sack_scores,
                    motion_tracker: SackMotionTracker,
                    ghosts: GhostSacks,
                    conf_tracker: ConfidenceTracker, cfg: dict,
                    ground_sacks: set = None):
    """
    Draw sack bounding boxes.

    ground_sacks: set of sack IDs that have been suppressed as floor sacks.
      These are drawn as a dim grey outline only — no label, no ownership colour.
      When a sack is picked up again it leaves ground_sacks and resumes normal drawing.
    """
    if ground_sacks is None:
        ground_sacks = set()

    u = T.unit(frame)

    for (sid, x1, y1, x2, y2, cx, cy, sc) in raw_sacks:
        # ── Ground-suppressed sack: dim outline only, no label ──
        if sid in ground_sacks:
            continue

        still     = motion_tracker.is_still(sid)
        own_score = sack_scores.get(sid, 0.0)
        if still:
            color = C_SACK_ST
            lbl   = f"Sack {sid}"
        elif sid in sack_owner:
            conf_lbl = confidence_class(own_score, cfg)
            if own_score >= cfg["conf_class_high"]:
                color = C_OWNED
            elif own_score >= cfg["conf_class_medium"]:
                color = C_MED_CONF
            else:
                color = C_LOW_CONF
            lbl = f"Sack {sid} → W{sack_owner[sid]} · {conf_lbl.lower()}"
        else:
            color = C_SACK
            lbl   = f"Sack {sid}"
        T.outline(frame, x1, y1, x2, y2, color, T.px(1.5, u))
        T.text(frame, lbl, x1 + T.px(2, u), y1 - T.px(4, u),
               T.fs(9.5, u), color, 1, T.FONT_BODY)

    # Ghost overlay — only draw if ghost is owned.
    # Collect first so the (expensive) full-frame copy and the blend are
    # skipped entirely when there is nothing to draw.  The blend used to
    # pass 0.0 as the overlay weight, which discarded every ghost that had
    # just been drawn while still paying for the copy on every frame.
    visible = [
        (sid, g) for sid, g in ghosts.iter_ghosts()
        if g.get("owner") is not None and sid not in ground_sacks
    ]
    if not visible:
        return

    overlay = frame.copy()
    for sid, g in visible:
        gx1, gy1, gx2, gy2 = g["x1"], g["y1"], g["x2"], g["y2"]
        T.dashed_polygon(
            overlay,
            [(gx1, gy1), (gx2, gy1), (gx2, gy2), (gx1, gy2)],
            C_GHOST, T.px(1.5, u), dash=T.px(8, u), gap=T.px(6, u),
        )
        T.text(overlay, f"Sack {sid} · predicted", gx1 + T.px(2, u),
               gy1 - T.px(4, u), T.fs(9, u), C_GHOST, 1, T.FONT_BODY)
    cv2.addWeighted(overlay, _GHOST_ALPHA, frame, 1.0 - _GHOST_ALPHA, 0, frame)


# ─────────────────────────────────────────────────────────────
#  "What just happened" — a read-only projection of the logs
# ─────────────────────────────────────────────────────────────

def build_event_feed(delivery_log=None, box_delivery_log=None,
                     anomaly_log=None, reid_events=None,
                     src_fps: float = 20.0, limit: int = 8) -> list[dict]:
    """
    Fold the pipeline's existing logs into the design's event list.

    Reads only; nothing is recorded, deduplicated into, or removed from
    the logs it is given.  Each entry is
    ``{"frame", "time", "text", "tag", "style"}`` where *style* is one of
    the ``_TAG_*`` tuples.

    Args:
        delivery_log:     ``state["delivery_log"]`` — sack crossings.
        box_delivery_log: ``box_pipeline.delivery_log`` — box crossings.
        anomaly_log:      ``state["anomaly_log"]``.
        reid_events:      ``relinker.reid_events``.
        src_fps:          Source frame rate, used to render a timestamp.
        limit:            Newest N events to return.

    Returns:
        Newest-first list, at most *limit* long.
    """
    fps = max(float(src_fps or 0.0), 1e-3)
    events: list[dict] = []

    def stamp(fn: int) -> str:
        secs = int(fn / fps)
        return f"{secs // 60:d}:{secs % 60:02d}"

    # Only the tail of each log can survive the `limit` slice, so never
    # walk more of it than that.
    for rec in (delivery_log or [])[-limit:]:
        n   = int(rec.get("peak_count") or 1)
        pid = rec.get("person_id")
        events.append({
            "frame": rec.get("frame", 0),
            "text": f"Worker {pid} carried {n} sack"
                    f"{'s' if n != 1 else ''} through the door",
            "tag": "Counted", "style": _TAG_ACCENT,
        })
    for rec in (box_delivery_log or [])[-limit:]:
        events.append({
            "frame": rec.get("frame", 0),
            "text": f"Worker {rec.get('person_id')} carried a box "
                    f"through the door",
            "tag": "Counted", "style": _TAG_ACCENT,
        })
    for rec in (anomaly_log or [])[-limit:]:
        pid = rec.get("person_id")
        who = f"Worker {pid}" if pid is not None else "an unknown worker"
        events.append({
            "frame": rec.get("frame", 0),
            "text": f"Sack {rec.get('sack_id')} was given to {who} with low certainty",
            "tag": "Please review", "style": _TAG_OUTLINE,
        })
    for rec in (reid_events or [])[-limit:]:
        events.append({
            "frame": rec.get("frame", 0),
            "text": f"Worker {rec.get('old_id')} was briefly hidden, "
                    f"then recognised again",
            "tag": "Re-linked", "style": _TAG_NEUTRAL,
        })

    events.sort(key=lambda e: e["frame"], reverse=True)
    for event in events:
        event["time"] = stamp(event["frame"])
    return events[:limit]


# ─────────────────────────────────────────────────────────────
#  Console overlay
# ─────────────────────────────────────────────────────────────

def draw_count_block(frame, u, count_label, headline, stats=()):
    """
    Top-left: the one number that matters, then the supporting few.

    Args:
        count_label: Kicker over the headline ("Sacks in" / "Sacks out").
        headline:    The headline number.
        stats:       ``(kicker, value)`` pairs for the scrim strip beneath.

    Returns:
        ``(width, bottom_y)`` of the whole block.
    """
    pad_x = T.px(20, u)
    kick_scale = T.fs(10.5, u)
    num_scale  = T.fs(58, u)
    kick_h = T.text_h(kick_scale, 1, T.FONT_BODY)
    num_h  = T.text_h(num_scale, 3, T.FONT_HEAD)

    label   = count_label.upper()
    number  = str(headline)
    track   = max(1.0, 10.5 * u * 0.14)
    block_w = max(
        T.px(186, u),
        pad_x * 2 + T.text_w(label, kick_scale, 1, T.FONT_BODY, track),
        pad_x * 2 + T.text_w(number, num_scale, 3, T.FONT_HEAD),
    )
    top_pad, gap, bot_pad = T.px(9, u), T.px(8, u), T.px(11, u)
    block_h = top_pad + kick_h + gap + num_h + bot_pad

    T.fill(frame, 0, 0, block_w, block_h, T.ACCENT, 1.0)
    T.text(frame, label, pad_x, top_pad + kick_h, kick_scale,
           T.PAPER, 1, T.FONT_BODY, tracking=track)
    T.text(frame, number, pad_x, top_pad + kick_h + gap + num_h,
           num_scale, T.PAPER, 3, T.FONT_HEAD)

    if not stats:
        return block_w, block_h

    # Secondary stats on a scrim directly beneath — same column, same edge.
    stat_kick = T.fs(9.5, u)
    stat_num  = T.fs(21, u)
    sk_h = T.text_h(stat_kick, 1, T.FONT_BODY)
    sn_h = T.text_h(stat_num, 2, T.FONT_HEAD)
    strip_top = block_h
    strip_h   = T.px(8, u) + sk_h + T.px(5, u) + sn_h + T.px(8, u)
    col_gap = T.px(26, u)
    strip_w = pad_x * 2 - col_gap + sum(
        max(T.text_w(k.upper(), stat_kick, 1, T.FONT_BODY, 1.2),
            T.text_w(str(v), stat_num, 2, T.FONT_HEAD)) + col_gap
        for k, v in stats
    )
    strip_w = max(strip_w, block_w)
    T.fill(frame, 0, strip_top, strip_w, strip_top + strip_h,
           T.SCRIM, T.SCRIM_ALPHA)

    col_x = pad_x
    for kick, value in stats:
        T.kicker(frame, kick, col_x, strip_top + T.px(8, u) + sk_h, u,
                 T.NEUTRAL_400, size=9.5)
        T.text(frame, str(value), col_x,
               strip_top + T.px(8, u) + sk_h + T.px(5, u) + sn_h,
               stat_num, T.PAPER, 2, T.FONT_HEAD)
        col_x += max(
            T.text_w(kick.upper(), stat_kick, 1, T.FONT_BODY, 1.2),
            T.text_w(str(value), stat_num, 2, T.FONT_HEAD),
        ) + col_gap

    return strip_w, strip_top + strip_h


def draw_warning_card(frame, u, x, y, w, message):
    """The design's "Check this" callout — accent wash, accent left edge."""
    pad_x, pad_y = T.px(12, u), T.px(10, u)
    head_scale = T.fs(11, u)
    body_scale = T.fs(11.5, u)
    head_h = T.text_h(head_scale, 1, T.FONT_HEAD)
    body_h = T.text_h(body_scale, 1, T.FONT_BODY)
    inner  = w - pad_x * 2 - T.px(3, u)
    lines  = T.wrap(message, inner, body_scale, 1, T.FONT_BODY, max_lines=4)
    line_gap = int(body_h * 1.55)
    card_h = pad_y * 2 + head_h + T.px(6, u) + line_gap * len(lines)

    T.fill(frame, x, y, x + w, y + card_h, T.ACCENT_200, 0.94)
    T.edge(frame, x, y, y + card_h, T.ACCENT, T.px(3, u))

    tx = x + T.px(3, u) + pad_x
    T.text(frame, "CHECK THIS", tx, y + pad_y + head_h, head_scale,
           T.ACCENT_900, 1, T.FONT_HEAD, tracking=max(1.0, 11 * u * 0.06))
    by = y + pad_y + head_h + T.px(6, u) + body_h
    for line in lines:
        T.text(frame, line, tx, by, body_scale, T.ACCENT_900, 1, T.FONT_BODY)
        by += line_gap
    return card_h


def _draw_workers_panel(frame, u, x, y, w, workers):
    """\"Counting now\" — one meter per worker, share of the session total."""
    pad_x, pad_y = T.px(11, u), T.px(9, u)
    kick_scale = T.fs(10, u)
    row_scale  = T.fs(11.5, u)
    kick_h = T.text_h(kick_scale, 1, T.FONT_BODY)
    row_h  = T.text_h(row_scale, 1, T.FONT_BODY)

    row_block = row_h + T.px(7, u) + T.px(4, u) + T.px(11, u)
    panel_h = pad_y * 2 + kick_h + T.px(8, u) + row_block * max(len(workers), 1)

    T.fill(frame, x, y, x + w, y + panel_h, T.SCRIM, T.SCRIM_ALPHA)
    T.kicker(frame, "Counting now", x + pad_x, y + pad_y + kick_h, u,
             T.NEUTRAL_400, size=10)

    if not workers:
        T.text(frame, "No carriers confirmed yet",
               x + pad_x, y + pad_y + kick_h + T.px(8, u) + row_h,
               row_scale, T.NEUTRAL_500, 1, T.FONT_BODY)
        return panel_h

    top_total = max((w_["delivered"] for w_ in workers), default=0) or 1
    ry = y + pad_y + kick_h + T.px(8, u)
    inner = w - pad_x * 2
    for worker in workers:
        name = f"Worker {worker['id']}"
        line = (f"{worker['delivered']} sacks · {worker['boxes']} "
                f"box{'es' if worker['boxes'] != 1 else ''}")
        T.text(frame, name, x + pad_x, ry + row_h, row_scale,
               T.PAPER, 1, T.FONT_BODY)
        lw = T.text_w(line, row_scale, 1, T.FONT_BODY)
        T.text(frame, line, x + w - pad_x - lw, ry + row_h, row_scale,
               T.NEUTRAL_400, 1, T.FONT_BODY)
        # Carrying right now reads as an accent tick, not another number.
        if worker["carrying"]:
            T.fill(frame, x + pad_x - T.px(5, u), ry + T.px(1, u),
                   x + pad_x - T.px(2, u), ry + row_h, T.ACCENT, 1.0)
        T.meter(frame, x + pad_x, ry + row_h + T.px(7, u), inner, u,
                worker["delivered"] / top_total,
                track=T.PAPER, bar=T.ACCENT_500, height=4.0,
                track_alpha=0.16)
        ry += row_block
    return panel_h


def draw_stat_panel(frame, u, x, y, w, title, rows):
    """
    A scrim panel of label/value rows — the design's aside sections.

    Args:
        title: Uppercase kicker for the panel.
        rows:  ``(label, value)`` or ``(label, value, ink)`` tuples.
    """
    pad_x, pad_y = T.px(11, u), T.px(9, u)
    kick_scale = T.fs(10, u)
    row_scale  = T.fs(12, u)
    val_scale  = T.fs(15, u)
    kick_h = T.text_h(kick_scale, 1, T.FONT_BODY)
    row_h  = max(T.text_h(row_scale, 1, T.FONT_BODY),
                 T.text_h(val_scale, 2, T.FONT_HEAD))
    step   = row_h + T.px(9, u)
    panel_h = pad_y * 2 + kick_h + T.px(8, u) + step * max(len(rows), 1) \
        + T.px(4, u)

    T.fill(frame, x, y, x + w, y + panel_h, T.SCRIM, T.SCRIM_ALPHA)
    T.kicker(frame, title, x + pad_x, y + pad_y + kick_h, u,
             T.NEUTRAL_400, size=10)

    ry = y + pad_y + kick_h + T.px(8, u)
    for row in rows:
        label, value = row[0], row[1]
        ink = row[2] if len(row) > 2 else T.PAPER
        T.text(frame, label, x + pad_x, ry + row_h, row_scale,
               T.NEUTRAL_400, 1, T.FONT_BODY)
        vw = T.text_w(str(value), val_scale, 2, T.FONT_HEAD)
        T.text(frame, str(value), x + w - pad_x - vw, ry + row_h,
               val_scale, ink, 2, T.FONT_HEAD)
        ry += step
    return panel_h


def _draw_events_panel(frame, u, x, y, w, max_h, events):
    """"What just happened" — newest first, one tag each."""
    pad_x, pad_y = T.px(11, u), T.px(9, u)
    kick_scale = T.fs(10, u)
    time_scale = T.fs(10, u)
    body_scale = T.fs(11.5, u)
    kick_h = T.text_h(kick_scale, 1, T.FONT_BODY)
    body_h = T.text_h(body_scale, 1, T.FONT_BODY)

    time_w = T.px(34, u)
    text_x = x + pad_x + time_w
    inner  = x + w - pad_x - text_x
    if inner < T.px(60, u) or max_h < pad_y * 2 + kick_h:
        return 0

    header_h = pad_y + kick_h + T.px(8, u)
    line_gap = int(body_h * 1.5)

    # Measure first: the scrim is only as tall as the rows that fit.
    laid_out, used = [], header_h
    for event in events:
        lines = T.wrap(event["text"], inner, body_scale, 1,
                       T.FONT_BODY, max_lines=2)
        row_h = line_gap * len(lines) + T.px(6, u) + \
            T.text_h(T.fs(10, u), 1, T.FONT_BODY) + T.px(4, u) + T.px(9, u)
        if used + row_h > max_h:
            break
        laid_out.append((event, lines, row_h))
        used += row_h
    if not laid_out:
        return 0
    used += T.px(2, u)

    T.fill(frame, x, y, x + w, y + used, T.SCRIM, T.SCRIM_ALPHA)
    T.kicker(frame, "What just happened", x + pad_x, y + pad_y + kick_h, u,
             T.NEUTRAL_400, size=10)

    ry = y + header_h
    for event, lines, row_h in laid_out:
        T.text(frame, event["time"], x + pad_x, ry + body_h, time_scale,
               T.NEUTRAL_500, 1, T.FONT_BODY)
        ty = ry + body_h
        for line in lines:
            T.text(frame, line, text_x, ty, body_scale, T.PAPER,
                   1, T.FONT_BODY)
            ty += line_gap
        bg, ink, border = event["style"]
        T.chip(frame, event["tag"], text_x,
               ty - body_h + T.px(6, u), u,
               bg, ink, size=10.0, pad_x=7.0, pad_y=2.0, border=border)
        ry += row_h
        T.rule(frame, x + pad_x, ry - T.px(5, u), x + w - pad_x,
               T.PAPER, T.px(1, u), alpha=0.14)
    return used


def draw_status_strip(frame, u, running_label, fn, fps, elapsed, signal,
                       signal_ink=T.NEUTRAL_500, running=True):
    """Bottom-left mono footer — the design's nav status block."""
    pad_x, pad_y = T.px(14, u), T.px(9, u)
    label_scale = T.fs(11.5, u)
    meta_scale  = T.fs(10.5, u)
    label_h = T.text_h(label_scale, 1, T.FONT_BODY)
    meta_h  = T.text_h(meta_scale, 1, T.FONT_BODY)

    dot = T.px(8, u)
    meta = f"Frame {fn:,}   {fps:.1f} fps"
    if elapsed:
        meta += f"   {elapsed}"
    meta += f"   {_VERSION_TAG}"

    strip_w = pad_x * 2 + dot + T.px(7, u) + max(
        T.text_w(running_label, label_scale, 1, T.FONT_BODY),
        T.text_w(meta, meta_scale, 1, T.FONT_BODY),
        T.text_w(signal, meta_scale, 1, T.FONT_BODY) if signal else 0,
    )
    rows = 2 + (1 if signal else 0)
    strip_h = pad_y * 2 + label_h + (T.px(5, u) + meta_h) * (rows - 1)
    y0 = frame.shape[0] - strip_h

    T.fill(frame, 0, y0, strip_w, frame.shape[0], T.SCRIM, T.SCRIM_ALPHA)
    T.fill(frame, pad_x, y0 + pad_y + (label_h - dot) // 2,
           pad_x + dot, y0 + pad_y + (label_h - dot) // 2 + dot,
           T.ACCENT if running else T.NEUTRAL_500, 1.0)
    tx = pad_x + dot + T.px(7, u)
    T.text(frame, running_label, tx, y0 + pad_y + label_h, label_scale,
           T.PAPER, 1, T.FONT_BODY)
    ry = y0 + pad_y + label_h + T.px(5, u) + meta_h
    T.text(frame, meta, tx, ry, meta_scale, T.NEUTRAL_400, 1, T.FONT_BODY)
    if signal:
        T.text(frame, signal, tx, ry + T.px(5, u) + meta_h, meta_scale,
               signal_ink, 1, T.FONT_BODY)


def draw_hud(frame, confirmed,
             person_sack_del_counts=None, person_sack_delivery=None,
             box_del_counts=None, person_box_delivery=None,
             person_current_sacks=None, person_current_boxes=None,
             fps=0.0, fn=0, frame_stats=None, n_anomalies=0,
             n_relinks=0, n_reid=0, canonical_fn=None,
             total_sacks=None, total_boxes=None,
             count_label="Sacks in", events=None, warning=None,
             elapsed=None, running=True):
    """
    Draw the console overlay.

    The first block of parameters is unchanged.  Everything from
    *total_sacks* on is optional design chrome — omit it and the overlay
    falls back to what it can derive from the per-person dicts, which is
    what every existing caller gets.

    Args:
        total_sacks:  Session sack total for the headline number.
        total_boxes:  Session box total.
        count_label:  "Sacks in" / "Sacks out" — the headline's kicker.
        events:       Output of :func:`build_event_feed`.
        warning:      Text for the "Check this" callout, or None.
        elapsed:      Pre-formatted elapsed time for the footer.
        running:      False renders the footer as paused.
    """
    # Normalise: accept both old and new parameter names
    _sack_del    = person_sack_del_counts or person_sack_delivery or {}
    _box_del     = box_del_counts         or person_box_delivery  or {}
    _cur_sacks   = person_current_sacks   or {}
    _cur_boxes   = person_current_boxes   or {}
    _canon       = canonical_fn if canonical_fn is not None else (lambda x: x)
    _frame_stats = frame_stats or {}
    _n_relinks_total = n_relinks + n_reid

    u = T.unit(frame)
    fw = frame.shape[1]

    # ── Per-worker rows, richest first ────────────────────────
    workers = []
    for pid in sorted(confirmed):
        can = _canon(pid)
        workers.append({
            "id":        can,
            "delivered": _sack_del.get(can, 0),
            "boxes":     _box_del.get(can, 0),
            "carrying":  _cur_sacks.get(can, 0) + _cur_boxes.get(can, 0) > 0,
        })
    # Re-ID can map two raw pids onto one canonical worker; show it once.
    deduped, seen = [], set()
    for worker in workers:
        if worker["id"] in seen:
            continue
        seen.add(worker["id"])
        deduped.append(worker)
    workers = sorted(deduped, key=lambda w: (-w["delivered"], w["id"]))

    if total_sacks is None:
        total_sacks = sum(_sack_del.values())
    if total_boxes is None:
        total_boxes = sum(_box_del.values())

    # ── Headline block ────────────────────────────────────────
    draw_count_block(frame, u, count_label, total_sacks,
                     stats=(("Boxes", total_boxes), ("Workers", len(workers))))

    # ── Right-hand column ─────────────────────────────────────
    panel_w = int(min(max(fw * 0.20, T.px(190, u)), T.px(290, u)))
    panel_x = fw - panel_w
    cursor  = 0
    if warning:
        cursor += draw_warning_card(frame, u, panel_x, cursor, panel_w,
                                warning) + T.px(2, u)
    cursor += _draw_workers_panel(frame, u, panel_x, cursor, panel_w,
                                  workers[:6]) + T.px(2, u)
    if events:
        _draw_events_panel(frame, u, panel_x, cursor, panel_w,
                           int(frame.shape[0] * 0.92) - cursor, events)

    # ── Footer vitals ─────────────────────────────────────────
    signal_bits = [
        f"{_frame_stats.get('raw', 0)} seen",
        f"{_frame_stats.get('crossings', 0)} crossed",
    ]
    if n_anomalies > 0:
        signal_bits.append(f"{n_anomalies} to review")
    if _n_relinks_total > 0:
        signal_bits.append(f"{_n_relinks_total} re-linked")
    draw_status_strip(
        frame, u,
        "Counting" if running else "Paused",
        fn, fps,
        elapsed or datetime.now().strftime("%H:%M:%S"),
        "  ·  ".join(signal_bits),
        signal_ink=C_ANOMALY if n_anomalies > 0 else T.NEUTRAL_500,
        running=running,
    )
