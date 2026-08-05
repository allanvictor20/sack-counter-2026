"""
web/diagnostics.py — Diagnostic check runner.

Runs a YOLO model over every Nth frame of a source video and records a
per-sample time series (frame index, seconds in, detection count) so the
diagnostics page can render a *line chart* of detection activity over the
video, instead of just two totals.

The model is loaded lazily and only if ``ultralytics`` is importable.  If
it is not (e.g. running on a machine that only has the web console
installed), the runner still walks the video and produces zero-valued
samples so the chart renders correctly — the user sees the structure they
will get once a model is dropped in.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2


# ── Optional YOLO loader ──────────────────────────────────────
# Loaded once per process.  Stays None if ultralytics isn't installed
# or the model file is missing — the runner degrades to a zero-count
# walk in that case so the diagnostics page still produces a chart.
_MODEL_CACHE: dict[str, object] = {}


def _load_model(model_path: str):
    """Return a cached YOLO model, or None if it can't be loaded."""
    if not model_path:
        return None
    p = Path(model_path)
    if not p.exists():
        return None
    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception:
        _MODEL_CACHE[model_path] = None
        return None
    try:
        model = YOLO(str(p))
    except Exception:
        _MODEL_CACHE[model_path] = None
        return None
    _MODEL_CACHE[model_path] = model
    return model


class DiagnosticRunner:
    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._result = {}

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._result)

    def start(self, source: str, model_path: str, every: int = 30,
              cutoff: float = 0.35, mode: str = "enter"):
        with self._lock:
            if self._running:
                raise RuntimeError("Diagnostics are already running.")
            self._running = True
            self._result = {
                "status": "running", "source": source,
                "frames_checked": 0, "detections": 0,
                "every": every, "cutoff": cutoff, "mode": mode,
                "model_used": model_path or "",
                "samples": [],          # list of {frame, t, count}
                "started_at": time.time(),
            }

        thread = threading.Thread(
            target=self._run,
            args=(source, model_path, every, cutoff, mode),
            daemon=True
        )
        thread.start()

    def _run(self, source: str, model_path: str, every: int,
             cutoff: float, mode: str):
        try:
            # Resolve model: explicit path first, then per-mode fallback.
            target_model = Path(model_path) if model_path else None
            if target_model and not target_model.exists():
                target_model = None
            if target_model is None:
                fallback = Path("exit_best.pt") if mode == "exit" else Path("best.pt")
                if fallback.exists():
                    target_model = fallback
            model_str = str(target_model) if target_model else (model_path or "")
            model = _load_model(model_str) if target_model else None

            cap = cv2.VideoCapture(0 if source == "webcam" else source)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open: {source}")

            fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0) or 20.0
            frames_checked = 0
            detections = 0
            samples: list[dict] = []
            frame_idx = 0
            t0 = time.time()

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_idx % max(1, every) == 0:
                    frames_checked += 1
                    count = 0
                    if model is not None and frame is not None:
                        try:
                            results = model(frame, verbose=False)
                            # YOLO result.boxes.cls tells us what was
                            # detected; we count anything whose confidence
                            # is at or above the cutoff, regardless of
                            # class — the diagnostics page is about "did
                            # the model see anything", not which class.
                            for r in results:
                                confs = getattr(r.boxes, "conf", None)
                                if confs is None:
                                    continue
                                try:
                                    import torch as _t  # type: ignore
                                    if isinstance(confs, _t.Tensor):
                                        confs = confs.tolist()
                                except Exception:
                                    pass
                                for c in confs:
                                    if float(c) >= cutoff:
                                        count += 1
                        except Exception:
                            # A bad frame shouldn't kill the whole run;
                            # just record zero for this sample.
                            count = 0
                    detections += count
                    t = (frame_idx / fps) if fps else 0.0
                    samples.append({
                        "frame": frame_idx,
                        "t": round(t, 2),
                        "count": count,
                    })

                    # Publish a live snapshot every few samples so the
                    # polling client can watch the line grow.
                    if frames_checked % 3 == 0:
                        with self._lock:
                            self._result.update({
                                "frames_checked": frames_checked,
                                "detections": detections,
                                "samples": list(samples),
                                "model_used": model_str,
                            })

                frame_idx += 1

            cap.release()

            with self._lock:
                self._result = {
                    "status": "finished",
                    "source": source,
                    "frames_checked": frames_checked,
                    "detections": detections,
                    "samples": samples,
                    "model_used": model_str,
                    "every": every, "cutoff": cutoff, "mode": mode,
                    "started_at": t0,
                    "ended_at": time.time(),
                }

        except Exception as exc:
            with self._lock:
                self._result = {
                    "status": "error",
                    "error": str(exc),
                    "samples": self._result.get("samples", []),
                }
        finally:
            with self._lock:
                self._running = False


RUNNER = DiagnosticRunner()
