"""
history.py — Reads the JSON logs the pipeline already writes.

Every finished run leaves a file behind: ``delivery_log_v23.json`` for
entry mode, ``exit_log_<timestamp>.json`` for exit mode.  The History
and Report screens are views over those files — nothing here runs the
pipeline or writes anything.

The two log shapes differ (they were written by different modes and
never unified), so this module normalises them into one row shape the
templates can render without knowing which mode produced them.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def _read(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        # A half-written log from an interrupted run should drop out of
        # the list, not take the whole History screen down.
        return None


def _when(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime)


def list_sessions(root: str | Path = ".") -> list[dict]:
    """
    Every readable session log, newest first.

    Returns rows of ``{id, when, when_iso, source, mode, sacks, boxes,
    workers, tag, tone, path}``.
    """
    root = Path(root)
    rows: list[dict] = []

    for path in sorted(root.glob("delivery_log_*.json")):
        data = _read(path)
        if data is None:
            continue
        peak = data.get("person_peak_delivery") or {}
        anomalies = len(data.get("anomalies") or [])
        rows.append({
            "id":      path.name,
            "path":    str(path),
            "when":    _when(path),
            "source":  data.get("source") or "—",
            "mode":    "Sacks in",
            "sacks":   data.get("total_sacks", 0),
            "boxes":   data.get("total_boxes", 0),
            "workers": len(peak),
            "tag":     "review" if anomalies else "clean",
            "tone":    "outline" if anomalies else "accent",
        })

    for path in sorted(root.glob("exit_log_*.json")):
        data = _read(path)
        if data is None:
            continue
        flags = len(data.get("discrepancy_flags") or [])
        rows.append({
            "id":      path.name,
            "path":    str(path),
            "when":    _when(path),
            "source":  data.get("source") or "—",
            "mode":    "Sacks out",
            "sacks":   data.get("total_sacks_out", 0),
            "boxes":   0,
            "workers": 0,
            "tag":     "counts disagreed" if flags else "clean",
            "tone":    "outline" if flags else "accent",
        })

    rows.sort(key=lambda r: r["when"], reverse=True)
    for row in rows:
        # "%-d" is glibc-only, so zero-strip by hand for Windows.
        row["when_iso"] = row["when"].strftime("%d %b, %H:%M").lstrip("0")
    return rows


def load_report(path: str | Path) -> dict | None:
    """
    Build the Report screen's view model from one log file.

    Returns None if the file cannot be read, so the route can 404
    rather than render a page full of blanks.
    """
    path = Path(path)
    data = _read(path)
    if data is None:
        return None

    if "total_sacks_out" in data:
        return _exit_report(path, data)
    return _entry_report(path, data)


def _entry_report(path: Path, data: dict) -> dict:
    deliveries = data.get("deliveries") or []
    peak       = data.get("person_peak_delivery") or {}
    anomalies  = data.get("anomalies") or []
    analytics  = data.get("analytics") or {}

    sacks_by_worker: dict[str, int] = {}
    boxes_by_worker: dict[str, int] = {}
    trips_by_worker: dict[str, int] = {}
    conf_by_worker:  dict[str, list] = {}

    for rec in deliveries:
        pid = str(rec.get("person_id"))
        if rec.get("type") == "box":
            boxes_by_worker[pid] = boxes_by_worker.get(pid, 0) + 1
        else:
            sacks_by_worker[pid] = (sacks_by_worker.get(pid, 0)
                                    + int(rec.get("peak_count") or 1))
        trips_by_worker[pid] = trips_by_worker.get(pid, 0) + 1
        conf_by_worker.setdefault(pid, []).append(
            float(rec.get("ownership_confidence",
                          rec.get("confidence", 0.0)) or 0.0))

    workers = sorted(
        set(sacks_by_worker) | set(boxes_by_worker) | set(peak),
        key=lambda p: -(sacks_by_worker.get(p, 0) + boxes_by_worker.get(p, 0)),
    )

    rows = []
    for pid in workers:
        sacks = sacks_by_worker.get(pid, 0)
        boxes = boxes_by_worker.get(pid, 0)
        trips = trips_by_worker.get(pid, 0)
        confs = conf_by_worker.get(pid) or []
        conf  = sum(confs) / len(confs) if confs else 0.0
        rows.append({
            "who":   f"Worker {pid}",
            "sacks": sacks, "boxes": boxes, "total": sacks + boxes,
            "load":  f"{sacks / trips:.1f}" if trips else "—",
            "conf":  f"{conf:.3f}" if confs else "—",
            "tag":   _certainty(conf, sacks + boxes),
            "tone":  _certainty_tone(conf, sacks + boxes),
        })

    # Per-minute trend, bucketed from each delivery's frame number.
    fps = float(data.get("src_fps") or analytics.get("src_fps") or 20.0)
    per_minute: dict[int, int] = {}
    for rec in deliveries:
        if rec.get("type") == "box":
            continue
        minute = int(rec.get("frame", 0) / max(fps, 1.0) / 60)
        per_minute[minute] = (per_minute.get(minute, 0)
                              + int(rec.get("peak_count") or 1))
    trend = _trend(per_minute)

    certain = sum(1 for r in deliveries if r.get("confidence_class") == "HIGH")
    medium  = sum(1 for r in deliveries if r.get("confidence_class") == "MEDIUM")
    low     = len(deliveries) - certain - medium
    total   = max(len(deliveries), 1)

    return {
        "kind":   "enter",
        "id":     path.name,
        "source": data.get("source") or "—",
        "mode":   "counting sacks in",
        "when":   _when(path).strftime("%d %b %Y, %H:%M").lstrip("0"),
        "totals": [
            {"label": "Sacks counted in", "value": data.get("total_sacks", 0),
             "note": f"across {len(deliveries)} trips", "accent": True},
            {"label": "Boxes counted in", "value": data.get("total_boxes", 0),
             "note": f"across {sum(boxes_by_worker.values())} trips"},
            {"label": "Workers seen", "value": len(workers),
             "note": f"{sum(1 for r in rows if r['total'])} carried something"},
            {"label": "Needs a look", "value": len(anomalies),
             "note": "counted with less certainty"},
        ],
        "rows":  rows,
        "trend": trend,
        "trend_mean": (f"{sum(per_minute.values()) / len(per_minute):.1f}"
                       if per_minute else "0.0"),
        "trend_peak": max(per_minute.values()) if per_minute else 0,
        "trend_peak_minute": (max(per_minute, key=per_minute.get) + 1
                              if per_minute else 0),
        "confidence": [
            {"label": "Certain", "n": certain,
             "pct": f"{certain / total * 100:.0f}%", "fill": "accent",
             "note": "clear view, one owner throughout"},
            {"label": "Fairly certain", "n": medium,
             "pct": f"{medium / total * 100:.0f}%", "fill": "accent-400",
             "note": "brief blocking of the view"},
            {"label": "Review", "n": low,
             "pct": f"{max(low, 0) / total * 100:.0f}%", "fill": "neutral",
             "note": "owner changed or the track was lost"},
        ],
        "anomalies": [
            {"at": f"frame {a.get('frame', 0):,}",
             "text": (f"Sack {a.get('sack_id')} was given to Worker "
                      f"{a.get('person_id')} with an ownership score of "
                      f"{a.get('ownership_confidence')}. Open the video "
                      f"here if the totals look wrong.")}
            for a in anomalies[:12]
        ],
        "log_path": str(path),
    }


def _exit_report(path: Path, data: dict) -> dict:
    exit_log = data.get("exit_log") or []
    flags    = data.get("discrepancy_flags") or []
    fps      = float(data.get("src_fps") or 20.0)

    per_minute: dict[int, int] = {}
    for rec in exit_log:
        minute = int(rec.get("frame", 0) / max(fps, 1.0) / 60)
        per_minute[minute] = per_minute.get(minute, 0) + 1

    return {
        "kind":   "exit",
        "id":     path.name,
        "source": data.get("source") or "—",
        "mode":   "counting sacks out",
        "when":   _when(path).strftime("%d %b %Y, %H:%M").lstrip("0"),
        "totals": [
            {"label": "Sacks counted out", "value": data.get("total_sacks_out", 0),
             "note": "at the door", "accent": True},
            {"label": "Landed in the zone", "value": data.get("landing_exit_count", 0),
             "note": "the primary count"},
            {"label": "Reconciled", "value": data.get("reconciled_count", 0),
             "note": "the number to report"},
            {"label": "Counts disagreed", "value": len(flags),
             "note": "times during the session"},
        ],
        "rows":  [],
        "trend": _trend(per_minute),
        "trend_mean": (f"{sum(per_minute.values()) / len(per_minute):.1f}"
                       if per_minute else "0.0"),
        "trend_peak": max(per_minute.values()) if per_minute else 0,
        "trend_peak_minute": (max(per_minute, key=per_minute.get) + 1
                              if per_minute else 0),
        "confidence": [],
        "anomalies": [
            {"at": f"frame {f.get('frame', 0):,}",
             "text": (f"The door counted {f.get('door_count')} and the "
                      f"landing zone counted {f.get('landing_count')} — "
                      f"a difference of {f.get('diff')}.")}
            for f in flags[:12]
        ],
        "duration": f"{data.get('duration_seconds', 0) / 60:.1f} min",
        "log_path": str(path),
    }


def _trend(per_minute: dict[int, int]) -> list[dict]:
    """Bar data for the per-minute chart, gaps filled with zeroes."""
    if not per_minute:
        return []
    peak = max(per_minute.values()) or 1
    return [
        {"label": m + 1, "v": per_minute.get(m, 0),
         "h": round(per_minute.get(m, 0) / peak * 140),
         "peak": per_minute.get(m, 0) == peak}
        for m in range(max(per_minute) + 1)
    ]


def _certainty(conf: float, total: int) -> str:
    if total == 0:
        return "carried nothing"
    return ("certain" if conf >= 0.8 else
            "fairly certain" if conf >= 0.6 else "review")


def _certainty_tone(conf: float, total: int) -> str:
    if total == 0:
        return "neutral"
    return ("accent" if conf >= 0.8 else
            "neutral" if conf >= 0.6 else "outline")
