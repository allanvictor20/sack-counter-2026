"""
session.py — Runs a counting session in a background thread.

The pipeline is a blocking loop that owns a video capture and, in the
CLI, a display window.  The web console needs the same loop but with the
frames going to a browser instead, and with a stop button that does not
kill the process mid-write.

That is exactly what ``main.run(frame_sink=…, should_stop=…)`` provides,
so this module drives the real pipeline rather than reimplementing it.
Every number the browser shows is read from the live ``PipelineSession``
the loop is already updating.

Threading
---------
One session at a time (a second one would fight for the camera and the
GPU).  The worker thread owns the pipeline; this class owns a lock
around the small snapshot the HTTP handlers read.  The annotated frame
is JPEG-encoded once per publish, on the worker, so N viewers cost one
encode rather than N.
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import cv2

from ..drawing import build_event_feed


@dataclass
class SessionOptions:
    """Everything the Setup screen collects."""
    source:      str
    mode:        str   = "enter"          # "enter" | "exit"
    conf_sack:   float | None = None
    conf_person: float | None = None
    conf_box:    float | None = None
    save_output: bool  = False
    config_path: str | None = None
    use_door_crossing: bool = False

    def as_dict(self) -> dict:
        return {
            "source": self.source, "mode": self.mode,
            "conf_sack": self.conf_sack, "conf_person": self.conf_person,
            "conf_box": self.conf_box, "save_output": self.save_output,
            "use_door_crossing": self.use_door_crossing,
        }


@dataclass
class _Snapshot:
    """What the browser polls.  Plain data, safe to serialise."""
    status:   str  = "idle"      # idle | starting | running | finished | error
    mode:     str  = "enter"
    source:   str  = ""
    frame_no: int  = 0
    fps:      float = 0.0
    started:  float | None = None
    error:    str | None = None
    warning:  str | None = None
    totals:   dict = field(default_factory=dict)
    workers:  list = field(default_factory=list)
    events:   list = field(default_factory=list)
    summary:  dict | None = None


class SessionManager:
    """Owns at most one running session and the frame it last produced."""

    #: Publish at most this often, in seconds.  The pipeline runs as fast
    #: as it can; the browser does not need every frame, and JPEG-encoding
    #: each one would steal time from detection.
    PUBLISH_INTERVAL = 1.0 / 15.0

    def __init__(self) -> None:
        self._lock     = threading.Lock()
        self._thread:  threading.Thread | None = None
        self._stop     = threading.Event()
        self._snapshot = _Snapshot()
        self._jpeg:    bytes | None = None
        self._last_publish = 0.0
        self._options: SessionOptions | None = None

    # ── Lifecycle ─────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, options: SessionOptions) -> None:
        """
        Begin a session.  Raises if one is already running — the camera
        and the model cannot be shared, so refusing is better than
        silently producing two half-speed sessions.
        """
        if self.is_running:
            raise RuntimeError("A session is already running. Stop it first.")
        self._stop.clear()
        self._options = options
        with self._lock:
            self._snapshot = _Snapshot(status="starting", mode=options.mode,
                                       source=options.source,
                                       started=time.time())
            self._jpeg = None
        self._thread = threading.Thread(target=self._run, args=(options,),
                                        name="sack-counter-session",
                                        daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        """Ask the loop to finish.  It still writes its report and log."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    # ── Reads for the HTTP layer ──────────────────────────────

    def snapshot(self) -> dict:
        with self._lock:
            snap = self._snapshot
            elapsed = int(time.time() - snap.started) if snap.started else 0
            return {
                "status": snap.status, "mode": snap.mode,
                "source": snap.source, "frame": snap.frame_no,
                "fps": round(snap.fps, 1),
                "elapsed": f"{elapsed // 60}:{elapsed % 60:02d}",
                "error": snap.error, "warning": snap.warning,
                "totals": dict(snap.totals), "workers": list(snap.workers),
                "events": list(snap.events), "summary": snap.summary,
                "running": self.is_running,
                "has_frame": self._jpeg is not None,
            }

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    # ── Worker ────────────────────────────────────────────────

    def _run(self, options: SessionOptions) -> None:
        try:
            with self._lock:
                self._snapshot.status = "running"
            if options.mode == "exit":
                summary = self._run_exit(options)
            else:
                summary = self._run_enter(options)
            with self._lock:
                self._snapshot.status = "finished"
                self._snapshot.summary = summary
        except BaseException as exc:                 # noqa: BLE001
            # The browser is the only place this would otherwise surface,
            # so keep the message and the traceback for the server log.
            traceback.print_exc()
            with self._lock:
                self._snapshot.status = "error"
                self._snapshot.error = f"{type(exc).__name__}: {exc}"

    def _run_enter(self, options: SessionOptions) -> dict:
        from ..main import run
        return run(
            source      = options.source,
            conf_sack   = options.conf_sack,
            conf_person = options.conf_person,
            conf_box    = options.conf_box,
            save_output = options.save_output,
            headless    = True,               # the browser is the display
            config_path = options.config_path,
            frame_sink  = self._publish_enter,
            should_stop = self._stop.is_set,
        )

    def _run_exit(self, options: SessionOptions) -> dict:
        from ..exit.exit_main import run_exit_counter
        return run_exit_counter(
            source            = options.source,
            conf_sack         = options.conf_sack,
            save_output       = options.save_output,
            headless          = True,
            config_path       = options.config_path,
            use_door_crossing = options.use_door_crossing,
            frame_sink        = self._publish_exit,
            should_stop       = self._stop.is_set,
        )

    # ── Frame sinks ───────────────────────────────────────────

    def _should_publish(self) -> bool:
        now = time.time()
        if now - self._last_publish < self.PUBLISH_INTERVAL:
            return False
        self._last_publish = now
        return True

    def _encode(self, frame) -> bytes | None:
        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return buf.tobytes() if ok else None

    def _publish_enter(self, frame, session, frame_no: int, fps: float) -> None:
        if not self._should_publish():
            return
        state    = session.state
        relinker = session.relinker
        box_del  = session.box_pipeline.delivery_counts()

        workers, seen = [], set()
        for pid in state["confirmed_persons"]:
            can = relinker.canonical(pid)
            if can in seen:
                continue
            seen.add(can)
            workers.append({
                "id":        can,
                "delivered": len(state["person_sack_delivered"].get(can, set())),
                "boxes":     box_del.get(can, 0),
                "carrying":  state["person_load"].get(can, 0),
            })
        workers.sort(key=lambda w: (-w["delivered"], w["id"]))

        events = build_event_feed(
            delivery_log     = state["delivery_log"],
            box_delivery_log = session.box_pipeline.delivery_log,
            anomaly_log      = state["anomaly_log"],
            reid_events      = relinker.reid_events,
            src_fps          = session.src_fps,
            limit            = 12,
        )

        jpeg = self._encode(frame)
        with self._lock:
            snap = self._snapshot
            snap.frame_no = frame_no
            snap.fps      = fps
            snap.workers  = workers
            snap.events   = [
                {"time": e["time"], "text": e["text"], "tag": e["tag"],
                 "tone": _tone_for(e["tag"])}
                for e in events
            ]
            snap.totals = {
                "sacks":     state["total_sacks_counted"],
                "boxes":     session.box_pipeline.total_counted,
                "workers":   len(workers),
                "review":    state["n_anomalies"],
                "relinks":   len(relinker.reid_events),
                "label":     "Sacks in",
            }
            if jpeg is not None:
                self._jpeg = jpeg

    def _publish_exit(self, frame, state, frame_no: int, fps: float) -> None:
        if not self._should_publish():
            return
        jpeg = self._encode(frame)
        with self._lock:
            snap = self._snapshot
            snap.frame_no = frame_no
            snap.fps      = fps
            snap.workers  = []
            snap.totals = {
                "sacks":     state["total_sacks_out"],
                "landed":    state["landing_exit_count"],
                "pile":      state["landing_peak_count"],
                "pending":   len(state["tentative_crossings"]),
                "review":    len(state["discrepancy_flags"]),
                "label":     "Sacks out",
            }
            flags = state["discrepancy_flags"]
            snap.warning = (
                f"The door counted {flags[-1]['door_count']} sacks and the "
                f"landing zone counted {flags[-1]['landing_count']}. Check "
                f"the video around this point before trusting the total."
                if flags else None
            )
            snap.events = [
                {"time": f"frame {rec.get('frame', 0)}",
                 "text": f"Sack {rec.get('sack_id')} left the room",
                 "tag": "Counted", "tone": "accent"}
                for rec in state["exit_log"][-12:][::-1]
            ]
            if jpeg is not None:
                self._jpeg = jpeg


def _tone_for(tag: str) -> str:
    """Map an event tag onto one of the design's .tag variants."""
    if tag == "Counted":
        return "accent"
    if tag == "Please review":
        return "outline"
    return "neutral"


#: Process-wide session.  The app is a single-operator local tool, so one
#: module-level manager is the whole story — no registry, no session ids.
MANAGER = SessionManager()
