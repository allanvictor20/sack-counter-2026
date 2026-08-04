"""
diagnostics.py — The Detection check screen's data.

Runs YOLO and nothing else over a sampled set of frames, so the operator
can see what the model itself detects before any counting rule is
applied.  This is the same question ``diagnose_sacks.py`` answers from
the terminal; here the result is a view model instead of a printout, and
the sample frames come back as data URIs so the page is self-contained.

Deliberately samples rather than walking every frame: the point is to
show where the confidence scores cluster, and every Nth frame answers
that in seconds instead of minutes.
"""

from __future__ import annotations

import base64
import threading
import traceback

import cv2
import numpy as np

from .. import theme as T
from ..drawing import draw_count_block
from ..model_classes import resolve_class_indices


# Score buckets, matching the design's histogram x-axis.
_BUCKETS = [(round(0.05 + i * 0.1, 2)) for i in range(9)]


class DiagnosticsRunner:
    """Runs one detection check at a time, in a background thread."""

    def __init__(self) -> None:
        self._lock   = threading.Lock()
        self._thread: threading.Thread | None = None
        self._result: dict | None = None
        self._status = "idle"        # idle | running | done | error
        self._error: str | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, source: str, model_path: str, every: int = 30,
              cutoff: float = 0.35, max_samples: int = 6) -> None:
        if self.is_running:
            raise RuntimeError("A detection check is already running.")
        with self._lock:
            self._status = "running"
            self._error  = None
            self._result = None
        self._thread = threading.Thread(
            target=self._run,
            args=(source, model_path, every, cutoff, max_samples),
            name="sack-counter-diagnostics", daemon=True)
        self._thread.start()

    def snapshot(self) -> dict:
        with self._lock:
            return {"status": self._status, "error": self._error,
                    "result": self._result, "running": self.is_running}

    def _run(self, source, model_path, every, cutoff, max_samples) -> None:
        try:
            result = _detection_check(source, model_path, every, cutoff,
                                      max_samples)
            with self._lock:
                self._result = result
                self._status = "done"
        except BaseException as exc:                 # noqa: BLE001
            traceback.print_exc()
            with self._lock:
                self._status = "error"
                self._error  = f"{type(exc).__name__}: {exc}"


def _detection_check(source: str, model_path: str, every: int,
                     cutoff: float, max_samples: int) -> dict:
    from ultralytics import YOLO

    model    = YOLO(model_path)
    sack_cls = resolve_class_indices(model.names, required=("sack",),
                                     strict=("sack",))["sack"]

    cap = cv2.VideoCapture(0 if source == "webcam" else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 20.0
    fw           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scores: list[float] = []
    per_frame: list[dict] = []
    samples: list[dict] = []
    fn = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fn += 1
        if fn % every:
            continue

        results = model.predict(frame, conf=0.05, classes=[sack_cls],
                                verbose=False)
        confs, boxes = [], []
        if results and results[0].boxes is not None and len(results[0].boxes):
            b     = results[0].boxes
            confs = b.conf.cpu().numpy().tolist()
            boxes = b.xyxy.cpu().numpy().astype(int).tolist()

        scores.extend(confs)
        per_frame.append({"frame": fn, "n": len(confs)})

        if len(samples) < max_samples:
            samples.append({
                "caption": f"frame {fn:,} · {len(confs)} sacks",
                "data_uri": _annotate(frame, boxes, confs, fn, cutoff),
            })

    cap.release()

    return {
        "source": source, "model": model_path,
        "resolution": f"{fw}×{fh}",
        "total_frames": total_frames,
        "checked_frames": len(per_frame),
        "every": every, "cutoff": cutoff,
        "duration": f"{total_frames / max(fps, 1.0):.0f}s",
        "detections": len(scores),
        "histogram": _histogram(scores, cutoff),
        "sparkline": _sparkline(per_frame),
        "samples": samples,
        "zero_frames": sum(1 for p in per_frame if p["n"] == 0),
        "stats": _stats(scores),
        "kept": sum(1 for s in scores if s >= cutoff),
    }


def _annotate(frame, boxes, confs, fn: int, cutoff: float) -> str:
    """Draw the console's detection styling and return a data URI."""
    shot = frame.copy()
    u    = T.unit(shot)
    for (x1, y1, x2, y2), c in zip(boxes, confs):
        colour = (T.ACCENT if c >= cutoff else
                  T.ACCENT_400 if c >= cutoff * 0.6 else T.NEUTRAL_400)
        T.tracked_box(shot, x1, y1, x2, y2, u, colour, f"{c:.2f}",
                      label_ink=T.contrast_ink(colour), thickness=2.5)
    draw_count_block(shot, u, "Sacks seen", len(confs),
                     stats=(("Frame", fn), ("Cut-off", f"{cutoff:.2f}")))

    # Downscale before encoding: these are thumbnails in a 3-up grid, and
    # full-resolution JPEGs would make the page many megabytes.
    height, width = shot.shape[:2]
    scale = min(640 / max(width, 1), 1.0)
    if scale < 1.0:
        shot = cv2.resize(shot, (int(width * scale), int(height * scale)))
    ok, buf = cv2.imencode(".jpg", shot, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


def _histogram(scores: list[float], cutoff: float) -> list[dict]:
    if not scores:
        return []
    arr    = np.asarray(scores)
    counts = [int(((arr >= lo) & (arr < lo + 0.1)).sum()) for lo in _BUCKETS]
    peak   = max(counts) or 1
    return [
        {"label": f"{lo:.2f}", "v": n,
         "h": round(n / peak * 155),
         "kept": lo + 0.05 >= cutoff}
        for lo, n in zip(_BUCKETS, counts)
    ]


def _sparkline(per_frame: list[dict]) -> list[dict]:
    if not per_frame:
        return []
    peak = max((p["n"] for p in per_frame), default=0) or 1
    step = max(len(per_frame) // 60, 1)
    return [
        {"h": max(2, round(p["n"] / peak * 110)), "zero": p["n"] == 0}
        for p in per_frame[::step]
    ]


def _stats(scores: list[float]) -> dict:
    if not scores:
        return {"min": "—", "max": "—", "mean": "—", "median": "—"}
    arr = np.asarray(scores)
    return {"min": f"{arr.min():.3f}", "max": f"{arr.max():.3f}",
            "mean": f"{arr.mean():.3f}", "median": f"{np.median(arr):.3f}"}


#: Process-wide runner, same reasoning as the session manager.
RUNNER = DiagnosticsRunner()
