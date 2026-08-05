"""
app.py — The Sack Counter web console.

Serves the six screens of the console design and the endpoints behind
them.  The pipeline itself is untouched: sessions run through
``main.run`` / ``run_exit_counter`` with the frame-sink hook, so what the
browser shows is the same annotated frame the CLI window would show.

Run it:

    python -m sack_counter.web            # http://127.0.0.1:8000

Routes
------
GET  /                 Live
GET  /setup            Start a session
GET  /calibrate        Mark the doorway
GET  /report[/{id}]    Session report
GET  /history          Past sessions
GET  /diagnostics      Detection check

POST /api/session/start|stop      session lifecycle
GET  /api/session                 live snapshot (polled)
GET  /api/stream.mjpg             annotated frames
GET  /api/frame.jpg               one frame, for calibration
POST /api/calibration             save door polygon + room point, or landing zone
POST /api/settings                save confidence thresholds
POST /api/diagnostics/start       run a detection check
GET  /api/diagnostics             its result
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..config import load_config, save_calibration
from ..version import VERSION_TAG
from . import history as history_mod
from .diagnostics import RUNNER
from .session import MANAGER, SessionOptions

_HERE      = Path(__file__).parent
TEMPLATES  = Jinja2Templates(directory=str(_HERE / "templates"))

app = FastAPI(title=f"Sack Counter {VERSION_TAG}")
app.mount("/static", StaticFiles(directory=str(_HERE / "static")),
          name="static")


# ── Shared template context ───────────────────────────────────

def _shell(request: Request, screen: str, **extra) -> dict:
    snap = MANAGER.snapshot()
    return {
        "request": request, "screen": screen, "version": VERSION_TAG,
        "session": snap,
        "status_label": {
            "running":  "Counting", "starting": "Starting",
            "finished": "Finished", "error": "Stopped", "idle": "Idle",
        }.get(snap["status"], "Idle"),
        **extra,
    }


def _cfg(config_path: str | None = None) -> dict:
    return load_config(config_path)


# ── Screens ───────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def live(request: Request):
    cfg = _cfg()
    return TEMPLATES.TemplateResponse(request, "live.html", _shell(
        request, "live",
        calibrated=bool(cfg.get("door_polygon_points")),
    ))


@app.get("/setup", response_class=HTMLResponse)
def setup(request: Request):
    cfg = _cfg()
    door = cfg.get("door_polygon_points")
    landing = cfg.get("landing_zone_points")
    model = Path(cfg.get("model_path", "best.pt"))
    exit_model = Path(cfg.get("exit_model_path") or "exit_best.pt")
    checks = [
        {"ok": model.exists(), "label": "Model file found",
         "detail": f"{model} · {'ready' if model.exists() else 'not found'}"},
        {"ok": bool(door), "label": "Camera calibrated",
         "detail": (f"{len(door)} corners marked" if door
                    else "not calibrated yet")},
        {"ok": bool(landing), "label": "Landing zone marked",
         "detail": (f"{len(landing)} points marked" if landing
                    else "only needed when counting sacks out")},
        {"ok": True, "label": "Settings loaded",
         "detail": f"config.yaml · {len(cfg)} settings"},
        {"ok": exit_model.exists(), "label": "Floor-sack model",
         "detail": (f"{exit_model}" if exit_model.exists()
                    else "only needed when counting sacks out")},
    ]
    return TEMPLATES.TemplateResponse(request, "setup.html", _shell(
        request, "setup", cfg=cfg, checks=checks,
        can_start=bool(door) and model.exists(),
    ))


@app.get("/calibrate", response_class=HTMLResponse)
def calibrate(request: Request):
    cfg = _cfg()
    return TEMPLATES.TemplateResponse(request, "calibrate.html", _shell(
        request, "calibrate", cfg=cfg,
        existing=cfg.get("door_polygon_points") or [],
        room_point=cfg.get("door_room_point") or None,
        existing_landing=cfg.get("landing_zone_points") or [],
    ))


@app.get("/report", response_class=HTMLResponse)
def report_latest(request: Request):
    sessions = history_mod.list_sessions()
    if not sessions:
        return TEMPLATES.TemplateResponse(request, "report.html", _shell(
            request, "report", report=None))
    return RedirectResponse(f"/report/{sessions[0]['id']}", status_code=303)


@app.get("/report/{log_id}", response_class=HTMLResponse)
def report(request: Request, log_id: str):
    # Only ever read a log from the working directory — a log id is a
    # file name, never a path the caller can steer.
    safe = Path(log_id).name
    data = history_mod.load_report(Path(".") / safe)
    if data is None:
        raise HTTPException(404, f"No readable session log named {safe}")
    return TEMPLATES.TemplateResponse(request, "report.html", _shell(
        request, "report", report=data))


@app.get("/history", response_class=HTMLResponse)
def history(request: Request):
    return TEMPLATES.TemplateResponse(request, "history.html", _shell(
        request, "history", sessions=history_mod.list_sessions()))


@app.get("/api/log/{log_id}")
def download_log(log_id: str):
    """Serve a session log file as a downloadable attachment."""
    safe = Path(log_id).name
    p = Path(".") / safe
    if not p.exists() or not p.is_file():
        raise HTTPException(404, f"No log file named {safe}")
    return Response(
        content=p.read_bytes(),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{safe}"'},
    )


# Allowed video extensions for /api/upload (lowercase, with dot).
_UPLOAD_VIDEO_EXTS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv", ".flv", ".ts",
}


@app.post("/api/upload")
async def upload_video(request: Request, filename: str = ""):
    """Receive a video file picked from the browser's file picker, save it
    to the working directory, and return the local filename.

    Browsers hide the real path of files chosen via <input type="file">, so
    the only way to give the backend a video it can open is to upload the
    bytes and persist them next to the app. The returned ``path`` is what
    the existing source-input flows expect — a relative filename the
    backend can ``Path(...)``-resolve.
    """
    # Strip any client-side path components — keep only the basename.
    name = Path(filename).name if filename else ""
    if not name:
        # Fall back to a generic name if the client didn't send one.
        name = "upload.mp4"
    if Path(name).suffix.lower() not in _UPLOAD_VIDEO_EXTS:
        raise HTTPException(
            400,
            f"Only video files are accepted ({', '.join(sorted(_UPLOAD_VIDEO_EXTS))})",
        )
    target = Path(".") / name
    size = 0
    # Stream chunks straight to disk so large uploads don't blow up memory.
    with open(target, "wb") as out:
        async for chunk in request.stream():
            if chunk:
                out.write(chunk)
                size += len(chunk)
    return {"path": name, "size": size}


@app.get("/diagnostics", response_class=HTMLResponse)
def diagnostics(request: Request):
    cfg = _cfg()
    return TEMPLATES.TemplateResponse(request, "diagnostics.html", _shell(
        request, "diagnostics", cfg=cfg, diag=RUNNER.snapshot()))


# ── Session API ───────────────────────────────────────────────

@app.post("/api/session/start")
async def api_session_start(request: Request):
    body = await request.json()
    source = (body.get("source") or "").strip()
    if not source:
        raise HTTPException(400, "Choose a video file or the webcam first.")
    if source != "webcam" and not Path(source).exists():
        raise HTTPException(400, f"No such video file: {source}")

    cfg = _cfg()
    if not cfg.get("door_polygon_points"):
        raise HTTPException(
            400, "This camera is not calibrated yet — mark the doorway first.")

    mode = body.get("mode") or "enter"
    if mode == "exit" and not cfg.get("landing_zone_points"):
        raise HTTPException(
            400, "The landing zone isn't marked yet — counting sacks out "
                 "needs it calibrated first.")

    def _conf(key):
        value = body.get(key)
        return float(value) if value not in (None, "") else None

    options = SessionOptions(
        source            = source,
        mode              = mode,
        conf_sack         = _conf("conf_sack"),
        conf_person       = _conf("conf_person"),
        conf_box          = _conf("conf_box"),
        save_output       = bool(body.get("save_output")),
        use_door_crossing = bool(body.get("use_door_crossing")),
    )
    try:
        MANAGER.start(options)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"ok": True, "options": options.as_dict()}


@app.post("/api/session/stop")
def api_session_stop():
    MANAGER.stop()
    return {"ok": True, "session": MANAGER.snapshot()}


@app.get("/api/session")
def api_session():
    return JSONResponse(MANAGER.snapshot())


@app.get("/api/stream.mjpg")
def api_stream():
    """
    Multipart JPEG stream of the annotated frames.

    An <img> pointed at this URL is the whole Live view — no player, no
    codec negotiation, and it degrades to a still if the session ends.
    """
    boundary = "sackframe"

    def frames():
        blank_sent = False
        while True:
            jpeg = MANAGER.latest_jpeg()
            if jpeg is None:
                if blank_sent:
                    time.sleep(0.2)
                    continue
                blank_sent = True
                jpeg = _placeholder_jpeg()
            yield (b"--" + boundary.encode() + b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                   + jpeg + b"\r\n")
            time.sleep(1 / 15)

    return StreamingResponse(
        frames(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}")


@app.get("/api/frame.jpg")
def api_frame(source: str = "", at: int = 0):
    """
    One frame, for the calibration canvas.

    With no *source*, returns the running session's latest frame; with
    one, grabs frame *at* from that file so the doorway can be marked
    before any session starts.
    """
    if source:
        safe = source if source == "webcam" else str(Path(source))
        cap = cv2.VideoCapture(0 if safe == "webcam" else safe)
        if not cap.isOpened():
            raise HTTPException(400, f"Cannot open: {source}")
        if at:
            cap.set(cv2.CAP_PROP_POS_FRAMES, at)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise HTTPException(400, "Cannot read a frame from that source.")
        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return Response(buf.tobytes(), media_type="image/jpeg")

    jpeg = MANAGER.latest_jpeg() or _placeholder_jpeg()
    return Response(jpeg, media_type="image/jpeg")


@app.get("/api/source-info")
def api_source_info(source: str):
    """Width, height and frame count — the calibration canvas needs them."""
    cap = cv2.VideoCapture(0 if source == "webcam" else str(Path(source)))
    if not cap.isOpened():
        raise HTTPException(400, f"Cannot open: {source}")
    info = {
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps":    round(cap.get(cv2.CAP_PROP_FPS) or 20.0, 1),
    }
    cap.release()
    return info


# ── Calibration & settings ────────────────────────────────────

@app.post("/api/calibration")
async def api_calibration(request: Request):
    body   = await request.json()
    target = body.get("target") or "door"

    if target == "landing":
        pts = body.get("landing_points") or []
        if len(pts) < 3:
            raise HTTPException(
                400, "Mark at least three points around the landing area.")
        if len(pts) > 8:
            raise HTTPException(
                400, "That's too many points — eight is the most the "
                     "landing zone can have.")

        # Same reasoning as the door polygon below: validate the shape
        # before saving, so a degenerate polygon (e.g. collinear points)
        # is refused here rather than surfacing as a cryptic error at
        # the start of the next session.
        from ..exit.landing_zone import LandingZone
        try:
            zone = LandingZone(
                points=[(int(p[0]), int(p[1])) for p in pts])
        except ValueError as exc:
            raise HTTPException(400, f"{exc}") from exc

        cfg = _cfg()
        cfg["landing_zone_points"] = zone.points
        save_calibration(cfg)
        return {"ok": True, "target": "landing", "points": zone.points,
                "centroid": zone.centroid}

    points = body.get("points") or []
    room   = body.get("room_point")
    if len(points) != 4:
        raise HTTPException(400, "Mark exactly four corners of the doorway.")
    if not room:
        raise HTTPException(400, "Click one spot inside the room as well.")

    # Build the polygon before saving: DoorPolygon validates convexity,
    # and a degenerate outline should be refused here rather than at the
    # start of the next session.
    from ..door_polygon import DoorPolygon
    try:
        door = DoorPolygon(points=[(int(p[0]), int(p[1])) for p in points],
                           room_point=(int(room[0]), int(room[1])))
    except (AssertionError, ValueError) as exc:
        raise HTTPException(400, f"{exc}") from exc

    cfg = _cfg()
    cfg["door_polygon_points"] = door.points
    cfg["door_room_point"]     = door.room_point
    save_calibration(cfg)
    return {"ok": True, "target": "door", "points": door.points,
            "room_point": door.room_point, "centroid": door.centroid}


@app.post("/api/settings")
async def api_settings(request: Request):
    body = await request.json()
    cfg  = _cfg()
    for key in ("conf_sack", "conf_person", "conf_box"):
        if body.get(key) is not None:
            cfg[key] = float(body[key])
    save_calibration(cfg)
    return {"ok": True, "cfg": {k: cfg[k] for k in
                                ("conf_sack", "conf_person", "conf_box")}}


# ── Detection check ───────────────────────────────────────────

@app.post("/api/diagnostics/start")
async def api_diagnostics_start(request: Request):
    body   = await request.json()
    source = (body.get("source") or "").strip()
    if not source:
        raise HTTPException(400, "Choose a video to check first.")
    cfg = _cfg()
    mode = body.get("mode") or "enter"
    try:
        RUNNER.start(
            source     = source,
            model_path = cfg.get("model_path", "best.pt"),
            every      = int(body.get("every") or 30),
            cutoff     = float(body.get("cutoff") or cfg.get("conf_sack", 0.35)),
            mode       = mode,
        )
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"ok": True}


@app.get("/api/diagnostics")
def api_diagnostics():
    return JSONResponse(RUNNER.snapshot())


# ── Helpers ───────────────────────────────────────────────────

_PLACEHOLDER: bytes | None = None


def _placeholder_jpeg() -> bytes:
    """A dark frame for when no session has produced one yet."""
    global _PLACEHOLDER
    if _PLACEHOLDER is None:
        import numpy as np
        from .. import theme as T
        frame = np.zeros((900, 1600, 3), np.uint8)
        frame[:] = T.NEUTRAL_900
        u = T.unit(frame)
        msg = "No session running"
        scale = T.fs(22, u)
        width = T.text_w(msg, scale, 1, T.FONT_HEAD)
        T.text(frame, msg, (1600 - width) // 2, 450, scale,
               T.NEUTRAL_600, 1, T.FONT_HEAD)
        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        _PLACEHOLDER = buf.tobytes() if ok else b""
    return _PLACEHOLDER