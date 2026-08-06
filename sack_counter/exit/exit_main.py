"""
exit_main.py — Main processing loop for exit sack counting mode.

Counting strategy — landing zone ONLY
----------------------------------------
Every raw detection is filtered against the landing zone polygon
BEFORE it reaches track confirmation. A sack detected anywhere else
in the frame — near the door, mid-corridor, still in someone's hands
— is discarded outright: it gets no track ID state, no hit/miss
counters, no confirmed_sacks entry. It does not exist to the pipeline
at all until it physically overlaps the landing zone.

This means --use-door-crossing is now a no-op: door-crossing depends
on sacks having tracked state near the door polygon, which can no
longer happen. total_sacks_out will stay 0 regardless of that flag.
A warning is printed at startup if it's passed.

Door calibration is still offered (when not headless) purely as a
VISUAL REFERENCE — so you can see where the door is while drawing the
landing zone polygon — it plays no role in counting.

Pipeline per frame
---------------------
1. YOLO detects sacks (class 1 only — persons skipped).
2. ByteTrack assigns stable IDs.
3. Detections are filtered to landing-zone-overlap only.
4. update_exit_sacks() confirms/evicts tracks (zone-filtered detections only).
5. update_landing_zone() tracks the landing zone accumulator (SOLE counter).
6. draw_exit_frame() renders overlays.

No person detection, no carrier stamping, no peak windows.
"""

from __future__ import annotations

import time
import logging
import numpy as np
import cv2
from collections import deque

from ..console import force_utf8_stdio
from ..config import load_config, save_calibration
from ..session_view import SessionView
from ..model_classes import resolve_class_indices
from ..door_polygon import calibrate_door_polygon, DoorPolygon, _get_display_cap
from .landing_zone import calibrate_landing_zone, LandingZone
from .exit_state import ExitPipelineState
from .exit_tracker import update_exit_sacks, cleanup_stale_tentatives
from .exit_crossing import (
    update_exit_crossings,
    confirm_tentative_crossings,
    expire_old_projections,
)
from .exit_landing import (
    make_landing_zone_tracker,
    update_landing_zone,
    check_discrepancy,
)
from .exit_drawing import draw_exit_frame
from .exit_reporter import print_exit_report, save_exit_log

logger = logging.getLogger(__name__)

# Discrepancy check interval (frames)
_DISCREPANCY_CHECK_INTERVAL = 150   # every ~5 s at 30 fps


def run_exit_counter(
    source:           str,
    conf_sack:        float | None = None,
    save_output:      bool         = False,
    headless:         bool         = False,
    config_path:      str | None   = None,
    use_door_crossing: bool        = False,
    frame_sink:       "callable | None" = None,
    should_stop:      "callable | None" = None,
) -> dict:
    """
    Run the exit sack counter on *source*.

    Args:
        source:            Video file path or ``"webcam"``.
        conf_sack:         Override sack detection confidence threshold.
        save_output:       Write annotated video to ``exit_output.mp4``.
        headless:          Suppress the display window.
        config_path:       Path to YAML/JSON config file.
        frame_sink:        Optional ``fn(SessionView)`` called once per
                           frame after the overlay is drawn, so a host
                           can stream the annotated frames without a
                           display window.  Same contract as entry mode.
        should_stop:       Optional ``fn() -> bool`` polled each
                           iteration; True ends the run cleanly and the
                           report still covers the frames processed.
        use_door_crossing: Enable door-crossing as a secondary counting
                           signal alongside the landing zone (OFF by
                           default — landing zone alone is the primary
                           and, in practice, more reliable counter; see
                           module docstring for why door-crossing was
                           demoted from primary to optional).

    Returns:
        The session summary written to the exit log.
    """
    # ── Config ────────────────────────────────────────────────
    force_utf8_stdio()
    cfg = load_config(config_path)
    if conf_sack is not None:
        cfg["conf_sack"] = conf_sack

    # Every key below now lives in DEFAULT_CFG, so these read straight from
    # cfg.  The inline cfg.get(key, default) fallbacks they replace had
    # drifted out of step with config.yaml — door_cross_threshold_px in
    # particular defaulted to 20 here and 30 in entry mode, so the two modes
    # disagreed about where the door plane was.
    cross_threshold_px   = float(cfg["door_cross_threshold_px"])
    door_proximity_px    = float(cfg["door_entry_proximity_px"])
    confirm_timeout      = int(cfg["exit_confirm_timeout_frames"])
    dedup_radius_px      = float(cfg["exit_dedup_radius_px"])
    dedup_recency_window = int(cfg["exit_dedup_recency_window"])
    stale_tentative_frames = int(cfg["exit_stale_tentative_frames"])

    print(f"\n{'='*60}")
    print("  SACK EXIT COUNTER")
    print(f"  Source    : {source}")
    print(f"  Conf sack : {cfg['conf_sack']}")
    print(f"{'='*60}\n")

    # ── Model ─────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is not installed.  Run: pip install ultralytics"
        ) from exc

    # Exit mode prefers a dedicated model (exit_model_path) trained
    # specifically on ground-lying/piled sacks, since the entry-mode
    # model (model_path) was trained mostly on carried sacks and
    # performs poorly on exit-mode footage (confirmed via diagnostic:
    # median confidence ~0.10 on ground-lying sacks vs >0.35 typical
    # for carried sacks). Falls back to the shared model_path if no
    # dedicated exit model has been configured yet.
    exit_model_path = cfg.get("exit_model_path") or cfg["model_path"]
    model = YOLO(exit_model_path)
    print(f"  Using model: {exit_model_path}")
    # Resolve the sack class.  Exit mode cannot do anything without it, so
    # it is 'strict': a name match is required and a positional fallback is
    # refused, rather than silently counting whatever class 1 happens to be.
    sack_cls = resolve_class_indices(
        model.names, required=("sack",), strict=("sack",),
    )["sack"]
    print(f"  Sack class index: {sack_cls}")
    logger.info("Sack class index: %d", sack_cls)

    # ── Video source ───────────────────────────────────────────
    cap = cv2.VideoCapture(0 if source == "webcam" else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame from source.")
    fh, fw = first_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    # Track lifetimes are in seconds; convert with the source's real FPS.
    confirm_frames = max(2, int(src_fps * cfg["confirm_secs"]))
    miss_frames    = max(2, int(src_fps * cfg["miss_secs"]))

    # ── Calibration ───────────────────────────────────────────
    if not headless:
        print("  Step 1/2: Calibrate the DOOR polygon")
        door = _load_or_calibrate_door(cap, cfg)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        print("  Step 2/2: Calibrate the LANDING ZONE polygon")
        landing_zone = _load_or_calibrate_landing(cap, door, cfg)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Persist both polygons so --headless can replay this session.
        cfg["door_polygon_points"]  = door.points
        cfg["door_room_point"]      = door.room_point
        cfg["landing_zone_points"]  = landing_zone.points
        save_calibration(cfg)
    else:
        door         = _load_door_from_cfg(cfg)
        landing_zone = _load_landing_from_cfg(cfg)

    if use_door_crossing:
        print(
            "  Warning: --use-door-crossing has no effect. Detections are "
            "now filtered to the landing zone before tracking even begins, "
            "so no sack ever appears near the door with tracked state for "
            "the door-crossing path to act on. total_sacks_out will stay 0."
        )

    # ── State + tracker ────────────────────────────────────────
    # LandingZoneTracker now uses the SAME stillness-detection class
    # and config values (still_speed_thresh, still_speed_window) that
    # entry mode's ground-sack suppression already uses — this was the
    # actual bug behind sacks in the landing zone not being recognized:
    # exit mode had its own separate, untuned stillness threshold.
    state          = ExitPipelineState()
    motion_tracker = make_landing_zone_tracker(cfg)

    # ── Video writer ───────────────────────────────────────────
    out = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter("exit_output.mp4", fourcc, src_fps, (fw, fh))

    # ── FPS smoothing ──────────────────────────────────────────
    fps_history: deque = deque(maxlen=30)
    t_prev = time.perf_counter()

    fn            = 0
    total_frames  = 0

    # ── Display window setup (fit large video to screen) ─────
    # Same fix as the entry-mode main.py — see that file for full
    # explanation. Display is scaled to fit; saved/written frames
    # and screenshots stay at full original resolution.
    DISP_MAX_W, DISP_MAX_H = _get_display_cap()
    disp_scale = min(DISP_MAX_W / fw, DISP_MAX_H / fh, 1.0)  # never upscale
    disp_w, disp_h = int(fw * disp_scale), int(fh * disp_scale)
    WIN_NAME = "Sack Exit Counter"
    if not headless:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, disp_w, disp_h)
        cv2.moveWindow(WIN_NAME, 0, 0)

    print("  Running exit counter — press Q to quit\n")

    # ── Main loop ──────────────────────────────────────────────
    while True:
        if should_stop is not None and should_stop():
            break
        ret, frame = cap.read()
        if not ret:
            break

        fn           += 1
        total_frames += 1
        state["frame_no"] = fn

        # ── FPS ───────────────────────────────────────────────
        t_now = time.perf_counter()
        fps_history.append(1.0 / max(t_now - t_prev, 1e-6))
        fps    = float(np.mean(fps_history))
        t_prev = t_now

        # ── YOLO + ByteTrack (sack class only) ────────────────
        raw_detections = _detect_sacks(model, frame, cfg["conf_sack"], sack_cls)

        # ── Restrict to the landing zone ONLY ──────────────────
        # Per explicit requirement: sacks are not tracked, confirmed,
        # or counted unless they overlap the landing zone polygon.
        # Filtering here — before update_exit_sacks() ever sees them —
        # means a sack detected anywhere else in the frame (near the
        # door, mid-corridor, in someone's hands) gets no track state
        # at all: no sack_hit/sack_miss counters, no confirmed_sacks
        # entry, nothing. It simply does not exist to the pipeline
        # until it is physically inside the landing zone.
        raw_detections = [
            det for det in raw_detections
            if landing_zone.bbox_overlaps(
                det[1], det[2], det[3], det[4],
                min_fraction=float(cfg["exit_landing_min_overlap"]),
            )
        ]

        # ── Track confirmation ─────────────────────────────────
        confirmed_sacks = update_exit_sacks(
            state, raw_detections, fn,
            confirm_frames=confirm_frames,
            miss_frames=miss_frames,
        )
        active_ids = {s[0] for s in confirmed_sacks}

        # ── Door crossing detection (OPTIONAL — off by default) ─
        # See module docstring: landing zone alone is the primary,
        # more reliable counter. Door-crossing is now a secondary
        # signal only enabled via --use-door-crossing.
        if use_door_crossing:
            update_exit_crossings(
                state, door, fn,
                confirm_timeout=confirm_timeout,
                cross_threshold_px=cross_threshold_px,
                door_proximity_px=door_proximity_px,
            )

        # ── Landing zone (PRIMARY counter) ──────────────────────
        update_landing_zone(
            state, landing_zone, raw_detections, fn,
            motion_tracker=motion_tracker,
            dedup_radius_px=dedup_radius_px,
            dedup_recency_window=dedup_recency_window,
            min_overlap=float(cfg["exit_landing_min_overlap"]),
        )

        if use_door_crossing:
            # ── Confirm / cancel tentatives ─────────────────────
            confirm_tentative_crossings(state, fn, confirm_timeout=confirm_timeout)

            # ── Expire old projections ──────────────────────────
            expire_old_projections(state, active_ids)

            # ── Stale tentative safety net ──────────────────────
            cleanup_stale_tentatives(state, fn, max_age=stale_tentative_frames)

            # ── Periodic discrepancy check ──────────────────────
            if fn % _DISCREPANCY_CHECK_INTERVAL == 0:
                check_discrepancy(state, fn)

        # ── Draw ──────────────────────────────────────────────
        draw_exit_frame(
            frame, state, door, landing_zone, fps, fn,
            all_detections=raw_detections,
        )

        if save_output and out:
            out.write(frame)

        if frame_sink is not None:
            frame_sink(SessionView.for_exit(frame, state, fn, fps))

        if not headless:
            disp_frame = cv2.resize(frame, (disp_w, disp_h)) if disp_scale != 1.0 else frame
            cv2.imshow(WIN_NAME, disp_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv2.imwrite(f"exit_screenshot_{fn}.jpg", frame)
                print(f"  Screenshot saved: exit_screenshot_{fn}.jpg")

    # ── Teardown ───────────────────────────────────────────────
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    # Final discrepancy check (only meaningful if door-crossing is
    # active — otherwise total_sacks_out stays 0 forever by design and
    # would falsely flag a discrepancy against the landing zone count
    # every single time)
    if use_door_crossing:
        check_discrepancy(state, fn)

    print_exit_report(state, source, total_frames, src_fps,
                      use_door_crossing=use_door_crossing)
    log_path = save_exit_log(state, source, total_frames, src_fps)

    # Returned so a host can report on the run without re-reading the
    # file it just wrote.  The CLI ignores this.
    return {
        "total_sacks_out":    state["total_sacks_out"],
        "landing_exit_count": state["landing_exit_count"],
        "exit_log":           state["exit_log"],
        "discrepancy_flags":  state["discrepancy_flags"],
        "log_path":           log_path,
        "frames":             fn,
    }


# ── Private helpers ────────────────────────────────────────────

def _detect_sacks(
    model,
    frame: np.ndarray,
    conf_thresh: float,
    sack_cls: int,
) -> list[tuple]:
    """
    Run YOLO+ByteTrack on *frame* and return sack detections only.

    Returns:
        List of (track_id, x1, y1, x2, y2, cx, cy, conf).
    """
    results = model.track(
        frame,
        persist=True,
        conf=conf_thresh,
        classes=[sack_cls],
        verbose=False,
        tracker="bytetrack.yaml",
    )

    detections: list[tuple] = []
    if not results or results[0].boxes is None:
        return detections

    boxes = results[0].boxes
    if boxes.id is None:
        return detections

    ids   = boxes.id.cpu().numpy().astype(int)
    xyxy  = boxes.xyxy.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()

    for tid, (x1, y1, x2, y2), conf in zip(ids, xyxy, confs):
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        detections.append((int(tid), x1, y1, x2, y2, cx, cy, float(conf)))

    return detections


def _load_or_calibrate_door(cap, cfg: dict) -> DoorPolygon:
    """Always run interactive door-polygon calibration (non-headless runs).

    Like the landing zone, the door polygon is re-drawn on every
    interactive run rather than being silently reused from a saved
    ``calibration.yaml`` — a camera nudge between runs would otherwise
    go unnoticed. Headless mode is unaffected: it still requires
    ``door_polygon_points``/``door_room_point`` via ``_load_door_from_cfg``.
    """
    return calibrate_door_polygon(cap, cfg)


def _load_or_calibrate_landing(cap, door: DoorPolygon, cfg: dict) -> LandingZone:
    """Always run interactive landing-zone calibration (non-headless runs).

    Unlike the door polygon, the landing zone is re-drawn on every
    interactive run rather than being silently reused from a saved
    ``calibration.yaml`` — a landing area clicked for one delivery layout
    can be wrong for the next, and reusing it without the user noticing
    was worse than asking every time. Headless mode is unaffected: it
    still requires ``landing_zone_points`` via ``_load_landing_from_cfg``.
    """
    return calibrate_landing_zone(cap, door_polygon=door, cfg=cfg)


def _load_door_from_cfg(cfg: dict) -> DoorPolygon:
    """Load door polygon from config (headless mode — no fallback)."""
    pts = cfg.get("door_polygon_points")
    rpt = cfg.get("door_room_point")
    if not pts or not rpt:
        raise RuntimeError(
            "Headless exit mode requires door_polygon_points and "
            "door_room_point in config.yaml."
        )
    return DoorPolygon(points=[tuple(p) for p in pts], room_point=tuple(rpt))


def _load_landing_from_cfg(cfg: dict) -> LandingZone:
    """Load landing zone from config (headless mode — no fallback)."""
    pts = cfg.get("landing_zone_points")
    if not pts or len(pts) < 3:
        raise RuntimeError(
            "Headless exit mode requires landing_zone_points in config.yaml."
        )
    return LandingZone(points=[tuple(p) for p in pts])
