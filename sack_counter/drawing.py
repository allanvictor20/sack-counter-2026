"""
drawing.py — OpenCV drawing helpers for Sack Counter v13.

Functions:
    draw_person_box
    draw_box_detections
    draw_sack_boxes
    draw_hud
"""

import cv2
import numpy as np
from datetime import datetime

from .colors import (
    C_PERSON, C_SACK, C_SACK_ST, C_OWNED,
    C_GHOST, C_LINE_A, C_LINE_B, C_COUNTED,
    C_HUD_BG, C_ANOMALY, C_REID, C_BOX,
    C_LOW_CONF, C_MED_CONF,
)
from .confidence import ConfidenceTracker, confidence_class
from .trackers import SackMotionTracker, GhostSacks

# Muted grey used for ground-suppressed sacks
_C_GROUND = (90, 90, 90)


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
    color = C_REID if was_relinked else (C_COUNTED if counted else C_PERSON)
    # Thinner border — less visual bulk
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    display_pid = f"{canonical_pid}" + (f"(={pid})" if pid != canonical_pid else "")
    top_lbl = f"P#{display_pid}  {current_sacks}sk {current_boxes}bx"
    # Smaller font for the top label
    (tw, th), _ = cv2.getTextSize(top_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.rectangle(frame, (x1, y1 - th - 7), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, top_lbl, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
    # Delivered count kept larger — this is the number that matters
    del_lbl = f"S:{delivered_sacks}  B:{delivered_boxes}"
    (mw, mh), _ = cv2.getTextSize(del_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    mx = x1 + (x2 - x1 - mw) // 2
    cv2.rectangle(frame, (x1, y2 - mh - 10), (x2, y2), (0, 0, 0), -1)
    cv2.putText(frame, del_lbl, (mx, y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


# drawing.py  — replace draw_sack_boxes and draw_box_detections

def draw_box_detections(frame, raw_boxes, box_owner,
                        conf_tracker: ConfidenceTracker, cfg: dict,
                        ground_boxes: set = None):       # ← new param
    if ground_boxes is None:
        ground_boxes = set()
    for (bid, x1, y1, x2, y2, sc) in raw_boxes:
        if bid in ground_boxes:                          # ← skip unowned still boxes
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
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        conf_lbl = confidence_class(own_score, cfg)
        lbl = f"B#{bid}" + (f"→P{pid}[{conf_lbl}]" if pid else "")
        cv2.putText(frame, lbl, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)


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

    for (sid, x1, y1, x2, y2, cx, cy, sc) in raw_sacks:
        # ── Ground-suppressed sack: dim outline only, no label ──
        if sid in ground_sacks:
            continue

        still     = motion_tracker.is_still(sid)
        own_score = sack_scores.get(sid, 0.0)
        if still:
            color = C_SACK_ST
            lbl   = f"#{sid}"
        elif sid in sack_owner:
            conf_lbl = confidence_class(own_score, cfg)
            if own_score >= cfg["conf_class_high"]:
                color = C_OWNED
            elif own_score >= cfg["conf_class_medium"]:
                color = C_MED_CONF
            else:
                color = C_LOW_CONF
            lbl = f"#{sid}→P{sack_owner[sid]}[{conf_lbl}]"
        else:
            color = C_SACK
            lbl   = f"#{sid}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        # Smaller font for sack labels
        cv2.putText(frame, lbl, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)

    # Ghost overlay — only draw if ghost is owned
    overlay = frame.copy()
    for sid, g in ghosts.iter_ghosts():
        if g.get("owner") is None:
            continue
        if sid in ground_sacks:
            continue   # don't ghost-draw suppressed floor sacks
        gx1, gy1, gx2, gy2 = g["x1"], g["y1"], g["x2"], g["y2"]
        cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), C_GHOST, 1)
        cv2.putText(overlay, f"#{sid}~", (gx1 + 2, gy1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, C_GHOST, 1)
    cv2.addWeighted(overlay, 0.0, frame, 1.0, 0, frame)


def draw_hud(frame, confirmed,
             person_sack_del_counts=None, person_sack_delivery=None,
             box_del_counts=None, person_box_delivery=None,
             person_current_sacks=None, person_current_boxes=None,
             fps=0.0, fn=0, frame_stats=None, n_anomalies=0,
             n_relinks=0, n_reid=0, canonical_fn=None):
    # Normalise: accept both old and new parameter names
    _sack_del    = person_sack_del_counts or person_sack_delivery or {}
    _box_del     = box_del_counts         or person_box_delivery  or {}
    _cur_sacks   = person_current_sacks   or {}
    _cur_boxes   = person_current_boxes   or {}
    _canon       = canonical_fn if canonical_fn is not None else (lambda x: x)
    _frame_stats = frame_stats or {}
    _n_relinks_total = n_relinks + n_reid

    lines = [
        f"SACK COUNTER v20  {datetime.now().strftime('%H:%M:%S')}",
        f"FPS {fps:.1f}   frame {fn}",
        "",
        f"  Sacks raw   : {_frame_stats.get('raw', 0)}",
        f"  Gate cross  : {_frame_stats.get('crossings', 0)}",
    ]
    if n_anomalies > 0:
        lines.append(f"  Anomalies   : {n_anomalies}")
    if _n_relinks_total > 0:
        lines.append(f"  Re-IDs      : {_n_relinks_total}")
    lines.append("")
    for pid in sorted(confirmed):
        can   = _canon(pid)
        cs    = _cur_sacks.get(can, 0)
        cb    = _cur_boxes.get(can, 0)
        ds    = _sack_del.get(can, 0)
        db    = _box_del.get(can, 0)
        label = f"P#{can}" if can == pid else f"P#{can}(raw#{pid})"
        lines.append(f"  {label}  sk={cs}/{ds}  bx={cb}/{db}")
    if not confirmed:
        lines.append("  (no carriers confirmed yet)")

    # Tighter line spacing — reduces HUD height
    line_h  = 18
    box_h   = line_h * len(lines) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (6, 6), (400, 6 + box_h), C_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(lines):
        col = (C_REID    if "Re-IDs" in line and _n_relinks_total > 0 else
               C_ANOMALY if "Anomalies" in line and n_anomalies > 0 else
               C_COUNTED if "sk=" in line and any(
                   f"/{ds}" in line
                   for ds in _sack_del.values() if ds > 0
               ) else
               C_PERSON  if line.strip().startswith("P#") else
               (180, 180, 180))
        # Smaller HUD font
        cv2.putText(frame, line, (14, 24 + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)