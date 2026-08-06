"""
main.py — Main processing loop for Sack Counter v21.

This file contains only the ``run()`` entry point and the video loop.
All per-frame logic lives in dedicated pipeline modules:

    pipeline/person_tracker.py  — update_persons, timeout_persons
    pipeline/sack_tracker.py    — update_sacks
    pipeline/counting.py        — update_peak_counts
    pipeline/door_crossing.py   — update_door_zone, process_door_disappearance, check_door_reentry
    pipeline/reporter.py        — print_report
    pipeline/orchestrator.py    — PipelineSession (owns all components)
    pipeline/state_machine.py   — SackStateMachineRegistry
    pipeline/analytics.py       — AnalyticsEngine
    pipeline/ground_memory.py   — GroundMemory

All v17 bug fixes are retained (FIX-UNICODE, FIX-REENTRY, FIX-STAMP,
FIX-TTL, FIX-PEAK0, FIX 1-3b, BUG1-A/B, BUG2).
"""

from __future__ import annotations

import cv2
import json
import time
import numpy as np
from collections import deque

from .version import VERSION_TAG
from .console import force_utf8_stdio
from .session_log import session_log_path
from .session_view import SessionView
from .config import load_config, save_calibration
from .model_classes import resolve_class_indices, unmatched_classes
from .door_polygon import calibrate_door_polygon, DoorPolygon, _get_display_cap
from .detections import detection_conf_floor, parse_detections
from .drawing import (
    draw_sack_boxes, draw_box_detections, draw_person_box, draw_hud,
    build_event_feed,
)
from .pipeline import PipelineSession
from .pipeline.frame import advance_frame
from .pipeline.reporter import print_report


def run(
    source:      str,
    conf_sack:   float | None = None,
    conf_person: float | None = None,
    conf_box:    float | None = None,
    save_output: bool         = False,
    headless:    bool         = False,
    config_path: str | None   = None,
    frame_sink:  "callable | None" = None,
    should_stop: "callable | None" = None,
) -> dict:
    """
    Run the Sack Counter pipeline on *source*.

    Args:
        source:      Video file path or ``"webcam"``.
        conf_sack:   Override detection confidence for sacks.
        conf_person: Override detection confidence for persons.
        conf_box:    Override detection confidence for boxes.
        save_output: If True, writes annotated video to ``output_v23.mp4``.
        headless:    If True, suppresses the display window.
        config_path: Path to YAML or JSON config file.
        frame_sink:  Optional ``fn(SessionView)`` called once per frame
                     after the overlay is drawn.  Lets a host — the web
                     console — stream the annotated frames without a
                     display window.  ``view.frame`` is the live buffer;
                     copy it if you keep it.
        should_stop: Optional ``fn() -> bool`` polled at the top of each
                     iteration.  Returning True ends the run as cleanly
                     as reaching the end of the video, so the caller gets
                     a report for the frames that were processed.

    Returns:
        The session summary: totals, the log path, and the analytics
        block that was written to disk.
    """
    # ── Config ────────────────────────────────────────────────
    force_utf8_stdio()
    cfg = load_config(config_path)
    if conf_sack   is not None: cfg["conf_sack"]   = conf_sack
    if conf_person is not None: cfg["conf_person"]  = conf_person
    if conf_box    is not None: cfg["conf_box"]    = conf_box

    print(f"\n{'='*62}")
    print(f"  Sack-Per-Person Counter  {VERSION_TAG}")
    print(f"  Model       : {cfg['model_path']}  (0=person 1=sack 2=box)")
    print(f"  Source      : {source}")
    print(f"  Conf sack   : {cfg['conf_sack']}   "
          f"Conf person : {cfg['conf_person']}   "
          f"Conf box    : {cfg['conf_box']}")
    print(f"{'='*62}\n")

    # ── Model + video source ──────────────────────────────────
    # Imported here, not at module scope: importing this module should
    # not drag in torch, so the pipeline stages stay testable on their own.
    from ultralytics import YOLO

    model = YOLO(cfg["model_path"])
    print(f"Model classes: {model.names}")
    # Resolve by name rather than assuming 0=person/1=sack/2=box.  The
    # hardcoded indices were recorded only in a config comment, so a model
    # with a different class order counted the wrong objects silently.
    cls_idx = resolve_class_indices(model.names)

    # A positional fallback is the one startup condition that can make
    # every number on screen wrong, so the console says so in plain words
    # instead of leaving it in the scrollback.
    _fell_back = unmatched_classes(model.names)
    ui_warning = None
    if _fell_back:
        ui_warning = (
            "The model's class names could not be matched for "
            + ", ".join(_fell_back)
            + ", so the default order is being used. Confirm the counts "
              "look right before trusting them."
        )

    cap = cv2.VideoCapture(0 if source == "webcam" else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")
    ret, first = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame.")
    fh, fw = first.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    # ── Calibration (interactive) or headless ────────────────
    if headless:
        door_points = cfg.get("door_polygon_points")
        if door_points is None:
            raise RuntimeError(
                "Headless mode requires door_polygon_points in config. "
                "Run interactively once to calibrate, then save to config.yaml."
            )
        room_point = cfg.get("door_room_point")
        door = DoorPolygon(
            points=[tuple(p) for p in door_points],
            room_point=tuple(room_point) if room_point else None,
        )
    else:
        door = calibrate_door_polygon(cap, cfg)
        # Persist the calibration so --headless can actually replay it.
        # These assignments used to update only the in-memory cfg, so the
        # clicked polygon died with the process and headless mode could
        # never be satisfied without hand-editing YAML.
        cfg["door_polygon_points"] = door.points
        cfg["door_room_point"]     = door.room_point
        save_calibration(cfg)

    # ── Build session (owns all stateful components) ──────────
    session = PipelineSession(
        cfg=cfg, frame_width=fw, frame_height=fh,
        src_fps=src_fps,
    )
    session.set_door(door)
    state = session.state

    # DEBUG: sanity-check door orientation before running.
    print(f"[DOOR CHECK] room_point={door.room_point}  "
          f"normal_vec={door.normal_vec}")
    print(f"[DOOR CHECK] corridor_midpt={door.corridor_midpt}  "
          f"room_midpt={door.room_midpt}")
    if door.room_point is None:
        print("[DOOR CHECK] WARNING: room_point is None — normal_vec "
              "orientation is a guess (axes[1], arbitrary sign). "
              "Re-calibrate interactively and confirm door_room_point "
              "is saved to config.yaml.")

    print(f"  Source FPS       : {src_fps:.1f}")
    print(f"  Confirm after    : {session.confirm_frames} frames")
    print(f"  Miss timeout     : {session.miss_frames} frames")
    print(f"  Re-ID memory     : {session.reid_mem_frames} frames")
    print(f"  Ground suppress  : {session.suppress_frames} frames "
          f"({cfg['still_suppress_secs']}s)")
    print(f"  Door polygon : {session.door.points}")
    print(f"  Door centroid: {session.door.centroid}")
    print(f"  Door normal  : ({session.door.normal_vec[0]:.3f}, {session.door.normal_vec[1]:.3f})\n")

    # ── Optional video writer ─────────────────────────────────
    out = None
    if save_output:
        out = cv2.VideoWriter(
            f"output_{VERSION_TAG}.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            src_fps, (fw, fh),
        )

    fn       = 0
    fps_ring = deque(maxlen=30)
    t_prev   = time.time()
    t_start  = t_prev

    # ── Display window setup (fit large video to screen) ─────
    # cv2.imshow at native resolution opens a window bigger than the
    # screen on high-res footage, which crops to the top-left corner
    # and looks like an unwanted zoom.  Set up ONE resizable window
    # before the loop (not per-frame, for performance) and resize
    # each frame to fit it just before display.
    DISP_MAX_W, DISP_MAX_H = _get_display_cap()
    disp_scale = min(DISP_MAX_W / fw, DISP_MAX_H / fh, 1.0)  # never upscale
    disp_w, disp_h = int(fw * disp_scale), int(fh * disp_scale)
    WIN_NAME = f"Sack Counter {VERSION_TAG}"
    if not headless:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, disp_w, disp_h)
        cv2.moveWindow(WIN_NAME, 0, 0)

    # ════════════════════════════════════════════════════════
    #  Main loop
    # ════════════════════════════════════════════════════════
    while True:
        if should_stop is not None and should_stop():
            break
        ret, frame = cap.read()
        if not ret:
            break
        fn += 1
        state["frame"]    = frame
        state["_frame_no"] = fn
        session.analytics.update_frame(fn)

        # BUG1-A: clear eviction guard at top of every frame
        state["just_evicted_sacks"] = set()

        now = time.time()
        fps_ring.append(1.0 / max(now - t_prev, 1e-6))
        t_prev = now
        fps    = float(np.mean(fps_ring))

        # ── Detect, then run every pipeline stage ─────────────
        results = model.track(
            frame,
            persist=True,
            conf=detection_conf_floor(cfg),
            verbose=False,
        )
        detections = parse_detections(results, cls_idx, cfg)
        outcome    = advance_frame(session, detections, fn, src_fps)
        box_owner  = outcome["box_owner"]
        # ── Draw ──────────────────────────────────────────────
        _elapsed = int(now - t_start)
        _draw_frame(
            frame, session, box_owner, fps, fn,
            frame_stats=outcome["stats"],
            warning=ui_warning,
            elapsed=f"{_elapsed // 60}:{_elapsed % 60:02d}",
            src_fps=src_fps,
        )

        if save_output and out:
            out.write(frame)

        if frame_sink is not None:
            frame_sink(SessionView.for_entry(frame, session, fn, fps,
                                             warning=ui_warning))

        if not headless:
            disp_frame = cv2.resize(frame, (disp_w, disp_h)) if disp_scale != 1.0 else frame
            cv2.imshow(WIN_NAME, disp_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv2.imwrite(f"screenshot_{VERSION_TAG}_{fn}.jpg", frame)

    # ════════════════════════════════════════════════════════
    #  Teardown
    # ════════════════════════════════════════════════════════
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    analytics_summary = session.final_summary()

    print_report(
        delivery_log          = state["delivery_log"],
        box_pipeline          = session.box_pipeline,
        person_peak_count     = state["person_peak_delivery"],
        person_sack_delivered = state["person_sack_delivered"],
        total_sacks_counted   = state["total_sacks_counted"],
        relinker              = session.relinker,
        analytics_summary     = analytics_summary,
    )

    # Timestamped, not a constant name: a run must not overwrite the record
    # of the run before it.
    log_path = session_log_path("delivery_log")
    payload = {
        # Recorded so the History screen can say which video a run was of;
        # the log used to carry only the numbers, not what they were of.
        "source":               source,
        "mode":                 "enter",
        "src_fps":              src_fps,
        "total_frames":         fn,
        "total_sacks":          state["total_sacks_counted"],
        "total_boxes":          session.box_pipeline.total_counted,
        "reid_events":          session.relinker.reid_events,
        "deliveries":           state["delivery_log"] + session.box_pipeline.delivery_log,
        "anomalies":            state["anomaly_log"],
        "person_peak_delivery": state["person_peak_delivery"],
        "analytics":            analytics_summary,
    }
    with open(log_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Delivery log saved -> {log_path}\n")

    # Returned so a host can report on the run without re-reading the
    # file it just wrote.  The CLI ignores this.
    return {**payload, "log_path": log_path, "frames": fn}


# ─────────────────────────────────────────────────────────────
#  Drawing helper (kept here as it only calls drawing module)
# ─────────────────────────────────────────────────────────────

def _draw_frame(
    frame: np.ndarray,
    session: PipelineSession,
    box_owner: dict,
    fps: float,
    fn: int,
    frame_stats: dict,
    warning: str | None = None,
    elapsed: str | None = None,
    src_fps: float = 20.0,
) -> None:
    """
    Draw all overlays onto *frame* in-place.

    Args:
        frame:       BGR frame to annotate.
        session:     Active PipelineSession.
        box_owner:   Box-to-person assignment dict.
        fps:         Current smoothed frame rate.
        fn:          Current frame number.
        frame_stats: Counts dict (raw/boxes/still/assigned/crossings/ghosts).
        warning:     Text for the console's "Check this" callout, or None.
        elapsed:     Pre-formatted elapsed session time for the footer.
        src_fps:     Source frame rate, used to timestamp the event feed.
    """
    state            = session.state
    box_pipeline     = session.box_pipeline
    conf_tracker     = session.conf_tracker
    motion_tracker   = session.motion_tracker
    ghost_sacks      = session.ghost_sacks
    relinker         = session.relinker
    cfg              = session.cfg
    box_conf_tracker = session.box_conf_tracker

    session.door.draw(
        frame,
        count=state["total_sacks_counted"],
        label="DOOR",
        show_debug=True,
    )

    draw_sack_boxes(frame, state["raw_sacks"],
                    state["sack_owner"], state["sack_scores"],
                    motion_tracker, ghost_sacks, conf_tracker, cfg,
                    ground_sacks=state["ground_sacks"])

    draw_box_detections(frame, state["raw_boxes"],
                        box_owner, box_conf_tracker, cfg,
                        ground_boxes=state["ground_boxes"])

    box_del_counts = box_pipeline.delivery_counts()
    for pid in state["confirmed_persons"]:
        box_obj = state["person_boxes"].get(pid)
        if box_obj is None or pid in state["persons_past_door"]:
            continue
        can = relinker.canonical(pid)
        cs_val = state["person_load"].get(can, 0)
        cb_val = sum(1 for b, owner in box_owner.items() if owner == can)
        ds_val = len(state["person_sack_delivered"].get(can, set()))
        db_val = box_del_counts.get(can, 0)
        # Only draw a box around people who are actually carrying
        # something (a currently-held sack/box) or have delivered one
        # this session. Bystanders with nothing stamped to them are
        # skipped so the overlay isn't cluttered with empty-handed
        # people walking through the frame.
        if cs_val == 0 and cb_val == 0 and ds_val == 0 and db_val == 0:
            continue
        draw_person_box(
            frame, box_obj, pid, can,
            cs=cs_val,
            ds=ds_val,
            cb=cb_val,
            db=db_val,
            delivered=ds_val > 0,
            relinked=can != pid,
        )

    draw_hud(
        frame,
        state["confirmed_persons"],
        person_sack_del_counts={
            can: len(state["person_sack_delivered"].get(can, set()))
            for can in state["confirmed_persons"]
        },
        box_del_counts=box_del_counts,
        person_current_sacks={
            relinker.canonical(pid): state["person_load"].get(relinker.canonical(pid), 0)
            for pid in state["confirmed_persons"]
        },
        person_current_boxes={
            can: sum(1 for b, owner in box_owner.items() if owner == can)
            for can in state["confirmed_persons"]
        },
        fps=fps, fn=fn, frame_stats=frame_stats,
        n_anomalies=state["n_anomalies"],
        n_reid=len(relinker.reid_events),
        canonical_fn=relinker.canonical,
        total_sacks=state["total_sacks_counted"],
        total_boxes=box_pipeline.total_counted,
        count_label="Sacks in",
        warning=warning,
        elapsed=elapsed,
        events=build_event_feed(
            delivery_log     = state["delivery_log"],
            box_delivery_log = box_pipeline.delivery_log,
            anomaly_log      = state["anomaly_log"],
            reid_events      = relinker.reid_events,
            src_fps          = src_fps,
        ),
    )
