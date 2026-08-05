"""
web/history.py — Read and parse past session execution logs.

Two responsibilities:

1. ``list_sessions`` — produces the rows the history table iterates over
   (``id``, ``source``, ``mode``, ``when_iso``, ``sacks``, ``boxes``,
   ``workers``, ``tone``, ``tag``).  Every row always has every field,
   no matter how sparse the underlying log file is.

2. ``load_report`` — produces the rich dict the report page consumes:
   the basic summary, plus the structured sections the new report.html
   lays out (totals, per-worker rows / exit-log rows, per-minute trend,
   confidence bars, anomalies).  Sections that have no data are empty
   lists / ``None`` so the template's ``{% if … %}`` guards hide them.
"""

from __future__ import annotations

import json
from pathlib import Path


# ── Public API ────────────────────────────────────────────────────────

def list_sessions(log_dir: Path | str = ".") -> list[dict]:
    """List all session logs sorted by newest first."""
    p = Path(log_dir)
    logs = list(p.glob("delivery_log_*.json")) + list(p.glob("exit_log_*.json"))
    sessions = []

    for log_path in sorted(logs, key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["id"] = log_path.name
                data["log_path"] = str(log_path)
                sessions.append(_normalize_row(data))
        except (json.JSONDecodeError, OSError):
            continue

    return sessions


def load_report(log_path: Path) -> dict | None:
    """Load a specific session report log file and build the rich
    structure the report template expects."""
    if not log_path.exists():
        return None
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["id"] = log_path.name
            data["log_path"] = str(log_path)
            return _normalize_report(data)
    except (json.JSONDecodeError, OSError):
        return None


# ── Shared normalisation ──────────────────────────────────────────────

def _detect_mode(data: dict) -> str:
    mode_raw = str(data.get("mode", "")).lower()
    log_id = str(data.get("id", "")).lower()
    if "exit" in mode_raw or "out" in mode_raw or "exit_log_" in log_id:
        return "exit"
    return "enter"


def _ensure_summary(data: dict, mode: str) -> dict:
    """Pull every metric up to the summary dict so the report template
    can read either ``summary.x`` or the top-level ``x``."""
    summary = data.get("summary")
    if not isinstance(summary, dict):
        summary = {}

    for key in ("sacks_entered", "sacks_exited", "door_count",
                "landing_count", "landing_peak", "boxes", "workers",
                "duration_fmt", "duration_sec", "start_time", "end_time",
                "n_anomalies", "anomalies"):
        if key not in summary and key in data:
            summary[key] = data[key]

    summary["sacks_entered"] = int(summary.get("sacks_entered", 0) or 0)
    summary["sacks_exited"]  = int(summary.get("sacks_exited", 0)  or 0)
    summary["door_count"]    = int(summary.get("door_count", 0)    or 0)
    summary["landing_count"] = int(summary.get("landing_count", 0) or 0)

    if mode == "exit":
        summary["total_count"] = summary.get(
            "sacks_exited", summary["sacks_entered"])
        summary["display_label"] = "Sacks Exited"
    else:
        summary["total_count"] = summary["sacks_entered"]
        summary["display_label"] = "Sacks Entered"

    return summary


# ── History-row normalisation ────────────────────────────────────────

def _normalize_row(data: dict) -> dict:
    """Build the row the history table iterates over."""
    if not isinstance(data, dict):
        return _empty_row()

    mode = _detect_mode(data)
    data["mode"] = mode
    data["summary"] = _ensure_summary(data, mode)
    summary = data["summary"]

    sacks = int(summary.get("total_count", 0) or 0)
    boxes = int(summary.get("boxes", 0) or 0)
    workers = int(summary.get("workers", 0) or 0)
    review_raw = summary.get("n_anomalies",
                 summary.get("anomalies", 0))
    review = int(len(review_raw) if isinstance(review_raw, list)
                 else (review_raw or 0))

    data["when_iso"] = _format_when(data)
    data["sacks"]    = sacks
    data["boxes"]    = boxes
    data["workers"]  = workers
    data["review"]   = review

    if review > 0:
        data["tone"] = "outline"
        data["tag"]  = "Review"
    elif mode == "exit":
        data["tone"] = "accent-2"
        data["tag"]  = "Sacks out"
    else:
        data["tone"] = "accent"
        data["tag"]  = "Sacks in"

    return data


def _empty_row() -> dict:
    return {
        "id": "", "source": "", "mode": "enter",
        "when_iso": "", "sacks": 0, "boxes": 0, "workers": 0,
        "review": 0, "tone": "neutral", "tag": "",
        "summary": {},
    }


def _format_when(data: dict) -> str:
    for key in ("end_time", "start_time", "ended_at", "started_at", "timestamp"):
        val = data.get(key)
        if val:
            return str(val)
    return ""


# ── Report-page normalisation ────────────────────────────────────────

def _normalize_report(data: dict) -> dict:
    """Take a raw log dict and return the rich structure the report
    template consumes: ``totals``, ``rows``, ``exit_log``, ``trend``,
    ``confidence``, ``anomalies`` plus the basic metadata."""
    if not isinstance(data, dict):
        return {}

    mode = _detect_mode(data)
    data["mode"] = mode
    data["summary"] = _ensure_summary(data, mode)
    summary = data["summary"]

    data["when"]      = _format_when(data)
    data["when_iso"]  = data["when"]
    data["totals"]    = _build_totals(data, mode)
    data["rows"]      = _build_worker_rows(data)
    data["exit_log"]  = _build_exit_rows(data)
    data["trend"]     = _build_trend(data, mode)
    data["confidence"]= _build_confidence(data)
    data["anomalies"] = _build_anomalies(data, mode)

    # Trend aggregates the template prints next to the chart.
    if data["trend"]:
        values = [b.get("v", 0) for b in data["trend"]]
        peak_v = max(values) if values else 0
        mean_v = round(sum(values) / len(values), 1) if values else 0
        peak_idx = values.index(peak_v) if values else 0
        data["trend_mean"] = mean_v
        data["trend_peak"] = peak_v
        data["trend_peak_minute"] = peak_idx + 1
        # Mark the peak bar.
        if 0 <= peak_idx < len(data["trend"]):
            data["trend"][peak_idx]["peak"] = True

    return data


def _build_totals(data: dict, mode: str) -> list[dict]:
    """Four count tiles the report header shows."""
    s = data.get("summary", {})
    if mode == "exit":
        return [
            {"label": "Sacks Exited",  "value": s.get("sacks_exited", 0),
             "accent": True,  "note": "Sacks that left the room."},
            {"label": "Landing count", "value": s.get("landing_count", 0),
             "accent": False, "note": "Sacks counted in the landing zone."},
            {"label": "Door count",    "value": s.get("door_count", 0),
             "accent": False, "note": "Cross-checked at the door."},
            {"label": "Peak on floor", "value": s.get("landing_peak", 0),
             "accent": False, "note": "Most sacks resting at once."},
        ]
    return [
        {"label": "Sacks Entered", "value": s.get("sacks_entered", 0),
         "accent": True,  "note": "Sacks brought into the room."},
        {"label": "Door count",    "value": s.get("door_count", 0),
         "accent": False, "note": "Worker crossings at the door."},
        {"label": "Workers",       "value": s.get("workers", 0),
         "accent": False, "note": "Distinct carriers detected."},
        {"label": "Boxes",         "value": s.get("boxes", 0),
         "accent": False, "note": "Boxes delivered."},
    ]


def _build_worker_rows(data: dict) -> list[dict]:
    """Per-worker table for enter-mode reports.

    Reads any of the common shapes the pipeline might write:
    - ``workers`` as a list of dicts (already structured)
    - ``per_worker`` dict keyed by worker id
    - ``delivery_log`` flat list of events (aggregated here)
    """
    if not isinstance(data, dict):
        return []

    # Already-structured list of worker rows.
    workers = data.get("workers")
    if isinstance(workers, list) and workers and isinstance(workers[0], dict):
        rows = []
        for w in workers:
            sacks = int(w.get("sacks", w.get("delivered", 0)) or 0)
            boxes = int(w.get("boxes", 0) or 0)
            conf  = w.get("conf", w.get("confidence"))
            rows.append({
                "who":   str(w.get("id", w.get("who", "—"))),
                "sacks": sacks,
                "boxes": boxes,
                "total": int(w.get("total", sacks + boxes) or 0),
                "load":  w.get("load", w.get("avg_load", "—")),
                "conf":  f"{float(conf):.2f}" if _is_num(conf) else "—",
                "tone":  w.get("tone", "accent"),
                "tag":   w.get("tag",  "Counted"),
            })
        return rows

    # per_worker dict keyed by worker id.
    per_worker = data.get("per_worker")
    if isinstance(per_worker, dict) and per_worker:
        rows = []
        for wid, w in per_worker.items():
            if not isinstance(w, dict):
                continue
            sacks = int(w.get("sacks", w.get("delivered", 0)) or 0)
            boxes = int(w.get("boxes", 0) or 0)
            rows.append({
                "who":   str(wid),
                "sacks": sacks,
                "boxes": boxes,
                "total": int(w.get("total", sacks + boxes) or 0),
                "load":  w.get("load", w.get("avg_load", "—")),
                "conf":  f"{float(w['conf']):.2f}" if _is_num(w.get("conf")) else "—",
                "tone":  w.get("tone", "accent"),
                "tag":   w.get("tag",  "Counted"),
            })
        return rows

    # Flat delivery_log — aggregate by worker.
    log = data.get("delivery_log")
    if isinstance(log, list) and log:
        agg: dict[str, dict] = {}
        for e in log:
            if not isinstance(e, dict):
                continue
            wid = str(e.get("worker", e.get("person", e.get("who", "—"))))
            row = agg.setdefault(wid, {"sacks": 0, "boxes": 0})
            if e.get("kind") == "box" or e.get("box"):
                row["boxes"] += 1
            else:
                row["sacks"] += 1
        return [
            {"who": wid, "sacks": w["sacks"], "boxes": w["boxes"],
             "total": w["sacks"] + w["boxes"], "load": "—", "conf": "—",
             "tone": "accent", "tag": "Counted"}
            for wid, w in sorted(agg.items(),
                                 key=lambda kv: -kv[1]["sacks"])
        ]

    return []


def _build_exit_rows(data: dict) -> list[dict]:
    """Sacks-that-left table for exit-mode reports.

    Passes through ``exit_log`` if present, otherwise builds a thin
    list from ``summary`` so the table at least shows the count."""
    log = data.get("exit_log")
    if isinstance(log, list) and log:
        rows = []
        for r in log:
            if not isinstance(r, dict):
                continue
            tag = r.get("tag", "Counted")
            tone = r.get("tone")
            if not tone:
                tone = "outline" if r.get("discrepancy") else "accent"
            rows.append({
                "sack_id":       r.get("sack_id", r.get("id", "—")),
                "frame":         r.get("frame", r.get("at", "—")),
                "at":            r.get("at", r.get("frame", "—")),
                "door_count":    r.get("door_count"),
                "landing_count": r.get("landing_count"),
                "tone":          tone,
                "tag":           tag,
            })
        return rows

    # Fallback — fabricate one row from the totals.
    s = data.get("summary", {})
    n = int(s.get("sacks_exited", 0) or 0)
    if n == 0:
        return []
    return [{
        "sack_id": "—", "frame": "—", "at": "—",
        "door_count":    s.get("door_count"),
        "landing_count": s.get("landing_count"),
        "tone": "accent", "tag": "Counted",
    }]


def _build_trend(data: dict, mode: str) -> list[dict]:
    """Per-minute bar chart.

    Accepts ``trend`` as either:
    - already-structured: [{label, v, h}, ...]
    - a list of ints (sacks per minute)
    - a dict {minute: count}
    """
    t = data.get("trend")
    if isinstance(t, list) and t:
        if isinstance(t[0], dict):
            return [
                {"label": str(b.get("label", i + 1)),
                 "v":     int(b.get("v", 0) or 0),
                 "h":     int(b.get("h", b.get("v", 0) * 4) or 0)}
                for i, b in enumerate(t)
            ]
        # plain list of ints
        values = [int(v or 0) for v in t]
        peak = max(values) if values else 1
        return [
            {"label": str(i + 1), "v": v,
             "h": int((v / peak) * 120) if peak else 0}
            for i, v in enumerate(values)
        ]
    if isinstance(t, dict) and t:
        items = sorted(t.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0)
        values = [int(v or 0) for _, v in items]
        peak = max(values) if values else 1
        return [
            {"label": str(k), "v": v,
             "h": int((v / peak) * 120) if peak else 0}
            for k, v in items
        ]
    return []


def _build_confidence(data: dict) -> list[dict]:
    """Confidence bars for the report aside.

    Passes through ``confidence`` if structured, else builds three
    rows from the conf thresholds in the summary."""
    c = data.get("confidence")
    if isinstance(c, list) and c and isinstance(c[0], dict):
        return [
            {"label": row.get("label", ""),
             "n":     row.get("n", "—"),
             "pct":   row.get("pct", "0%"),
             "fill":  row.get("fill", ""),
             "note":  row.get("note", "")}
            for row in c
        ]

    s = data.get("summary", {})
    rows = []
    for key, label, note in (
        ("conf_sack",   "Sacks",  "Main counted object."),
        ("conf_person", "People", "Worker detections."),
        ("conf_box",    "Boxes",  "Box deliveries."),
    ):
        val = s.get(key) or data.get(key)
        if _is_num(val):
            pct = int(float(val) * 100)
            rows.append({
                "label": label, "n": f"{float(val):.2f}",
                "pct": f"{pct}%", "fill": "accent-400" if pct < 50 else "",
                "note": note,
            })
    return rows


def _build_anomalies(data: dict, mode: str) -> list[dict]:
    """Flagged events for the report aside.

    Prefers ``anomalies`` if structured, else falls back to
    ``anomaly_log`` (enter mode) or ``discrepancy_flags`` (exit mode)."""
    a = data.get("anomalies")
    if isinstance(a, list) and a and isinstance(a[0], dict):
        return [
            {"at":   str(r.get("at", r.get("time", r.get("frame", "—")))),
             "text": str(r.get("text", r.get("message", r.get("reason", ""))))}
            for r in a
        ]

    # Enter mode: anomaly_log is a list of dicts.
    log = data.get("anomaly_log")
    if isinstance(log, list) and log:
        return [
            {"at":   str(r.get("at", r.get("time", r.get("frame", "—")))),
             "text": str(r.get("text", r.get("reason", r.get("message", ""))))}
            for r in log if isinstance(r, dict)
        ]

    # Exit mode: discrepancy_flags carry door_count vs landing_count.
    flags = data.get("discrepancy_flags")
    if isinstance(flags, list) and flags:
        out = []
        for r in flags:
            if not isinstance(r, dict):
                continue
            door    = r.get("door_count", "?")
            landing = r.get("landing_count", "?")
            frame   = r.get("frame", r.get("at", "—"))
            out.append({
                "at":   f"frame {frame}",
                "text": (f"Door counted {door} sacks but the landing zone "
                         f"counted {landing}. Check the video around this "
                         f"point before trusting the total."),
            })
        return out

    return []


# ── Utils ─────────────────────────────────────────────────────────────

def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)