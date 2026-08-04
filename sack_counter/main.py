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
from .config import load_config, save_calibration
from .model_classes import resolve_class_indices, unmatched_classes
from .door_polygon import calibrate_door_polygon, DoorPolygon, _get_display_cap
from .assignment import assign_sacks_hungarian
from .drawing import (
    draw_sack_boxes, draw_box_detections, draw_person_box, draw_hud,
    build_event_feed,
)
from .pipeline import PipelineSession, SackState
from .pipeline.person_tracker import update_persons, timeout_persons
from .pipeline.sack_tracker import update_sacks
from .pipeline.counting import update_peak_counts
from .pipeline.door_crossing import update_door_zone, check_door_reentry
from .pipeline.reporter import print_report

from ultralytics import YOLO


def run(
    source:      str,
    conf_sack:   float | None = None,
    conf_person: float | None = None,
    conf_box:    float | None = None,
    save_output: bool         = False,
    headless:    bool         = False,
    config_path: str | None   = None,
) -> None:
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
    model = YOLO(cfg["model_path"])
    print(f"Model classes: {model.names}")
    # Resolve by name rather than assuming 0=person/1=sack/2=box.  The
    # hardcoded indices were recorded only in a config comment, so a model
    # with a different class order counted the wrong objects silently.
    cls_idx      = resolve_class_indices(model.names)
    CLS_PERSON   = cls_idx["person"]
    CLS_SACK     = cls_idx["sack"]
    CLS_BOX      = cls_idx["box"]

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
        cfg.setdefault("door_approach_side", "right")
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
    state  = session.state
    logger = session.logger

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

        # ── 1. Detection ──────────────────────────────────────
        results = model.track(
            frame,
            persist=True,
            conf=min(cfg["conf_sack"], cfg["conf_person"], cfg["conf_box"]),
            verbose=False,
        )
        raw_persons: list[tuple] = []
        raw_sacks:   list[tuple] = []
        raw_boxes:   list[tuple] = []

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is None:
                    continue
                cls = int(box.cls[0])
                sc  = float(box.conf[0])
                tid = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if cls == CLS_PERSON and sc >= cfg["conf_person"]:
                    raw_persons.append((tid, x1, y1, x2, y2, sc))
                elif cls == CLS_SACK and sc >= cfg["conf_sack"]:
                    raw_sacks.append((tid, x1, y1, x2, y2,
                                      (x1+x2)//2, (y1+y2)//2, sc))
                elif cls == CLS_BOX and sc >= cfg["conf_box"]:
                    raw_boxes.append((tid, x1, y1, x2, y2, sc))

        state["raw_sacks"] = raw_sacks
        state["raw_boxes"] = raw_boxes

        # ── 2. Person tracking ────────────────────────────────
        reid_before = len(session.relinker.reid_events)
        active_pids = update_persons(raw_persons, session, fn)
        timeout_persons(active_pids, session, fn)
        session.relinker.expire_lost(fn)
        # Feed relinks to analytics.  record_reid_event() existed but was
        # never called, so analytics.pipeline.reid_events reported 0 in the
        # same JSON file whose top-level reid_events list was populated.
        for _ in range(len(session.relinker.reid_events) - reid_before):
            session.analytics.record_reid_event()

        # ── 3. Sack tracking + ground suppression ─────────────
        update_sacks(raw_sacks, session, fn)

        # ── 4. Assignment (exclude ground sacks) ──────────────
        confirmed_sack_list = [
            s for s in raw_sacks
            if s[0] in state["confirmed_sacks"]
            and s[0] not in state["ground_sacks"]
        ]

        # Release stale ownership for sacks whose owner passed the gate
        for (sid, *_) in confirmed_sack_list:
            current_owner = session.ownership_mem.get(sid)
            if current_owner is not None and current_owner in state["persons_past_door"]:
                session.ownership_mem.clear(sid)
                state["sack_carrier_stamp"].pop(sid, None)
                logger.debug(
                    "STALE-OWNER release: sack#%d owner=P#%d cleared  frame=%d",
                    sid, current_owner, fn,
                )

        active_persons = {
            pid: box for pid, box in state["person_boxes"].items()
            if pid in state["confirmed_persons"]

        }

        sack_owner, sack_scores, person_load, assign_stats = assign_sacks_hungarian(
            confirmed_sack_list, active_persons,
            session.motion_tracker, session.ownership_mem, cfg,
            sack_embeddings=state["sack_embeddings"],
            person_emb_fn=lambda pid: state["person_embeddings"].get(pid),
            persons_past_gate=state["persons_past_door"],   # v21: exclude past-door persons
        )
        state["sack_owner"]  = sack_owner
        state["sack_scores"] = sack_scores
        state["person_load"] = person_load

        # Clean dicts: active sacks only (removes stale ghost-carried entries)
        _active_sids      = {s[0] for s in confirmed_sack_list}
        sack_owner_clean  = {s: p for s, p in sack_owner.items()  if s in _active_sids}
        sack_scores_clean = {s: v for s, v in sack_scores.items() if s in _active_sids}

        # Record ownership confidence + anomalies
        for sid, score in sack_scores.items():
            session.conf_tracker.record_ownership(sid, score)
            if score < cfg["ownership_anomaly_threshold"]:
                state["anomaly_log"].append({
                    "frame": fn, "type": "low_owner_confidence",
                    "sack_id": sid, "person_id": sack_owner.get(sid),
                    "ownership_confidence": round(score, 3),
                })
                state["n_anomalies"] = len(state["anomaly_log"])
                logger.warning(
                    "LOW OWNERSHIP CONF sack#%d -> P#%s score=%.3f  frame=%d",
                    sid, sack_owner.get(sid), score, fn,
                )
                session.analytics.record_anomaly()

        # ── 5. Box pipeline ───────────────────────────────────
        box_owner = session.box_pipeline.update(
            raw_boxes, active_persons, fn, src_fps, cfg=cfg,
        )

        # Box ground suppression (visual only)
        active_bids = {b[0] for b in raw_boxes}
        for (bid, *_) in raw_boxes:
            if box_owner.get(bid) is None:
                state["ground_boxes"].add(bid)
            else:
                state["ground_boxes"].discard(bid)
        state["ground_boxes"] &= active_bids

        # ── 6. Peak counts ────────────────────────────────────
        if cfg.get("peak_count_enabled", True):
            update_peak_counts(
                confirmed_sack_list, session, fn,
                sack_owner_clean, sack_scores_clean,
            )

        
        # ── 7a. Re-entry reset ────────────────────────────────
        check_door_reentry(session, fn)

        # ── 7b. Door crossing commits ─────────────────────────
        # Pass active_pids so update_door_zone can distinguish a genuinely
        # missing person from one whose stale person_prev_cx looks valid.
        crossings_before = len(state["delivery_log"])
        update_door_zone(session, fn, active_pids)
        n_crossings = len(state["delivery_log"]) - crossings_before

        
        # ── 8. Peak state cleanup (persons long past door) ────
        # FIX: this used to compare raw pid_cx against door_cx using a
        # left/right heuristic (door_approach_side) — leftover from
        # before the v22 normal-vector rewrite. Every other geometry
        # check in the pipeline (crossing, window, direction) uses
        # door.project_onto_normal(); this one didn't, so for any door
        # that isn't near-perfectly horizontal/vertical the threshold
        # didn't correspond to the real corridor/room boundary, and could
        # wipe person_peak_count for a carrier who was still mid-approach
        # (this loop had no persons_past_door guard at all).
        if cfg.get("peak_count_enabled", True):
            stale_margin = cfg["peak_freeze_px"] * 2
            for pid in list(state["person_peak_count"].keys()):
                if pid not in state["persons_past_door"]:
                    continue
                pid_cx = state["person_prev_cx"].get(pid)
                if pid_cx is None:
                    continue
                pid_cy = state["person_boxes"].get(pid)
                cy_val = ((pid_cy[1] + pid_cy[3]) // 2) if pid_cy else 0
                proj = session.door.project_onto_normal(pid_cx, cy_val)
                if proj > stale_margin:
                    state["person_peak_count"].pop(pid, None)
                    state["person_approach_fn"].pop(pid, None)

        # ── 9. Expire stale orphaned peaks (BUG2) ─────────────
        expired = [pid for pid, rec in state["orphaned_peaks"].items()
                   if fn > rec["expires_at"]]
        for pid in expired:
            state["orphaned_peaks"].pop(pid)
            logger.debug("Orphaned peak P#%d expired without commit  frame=%d", pid, fn)
        if expired:
            active_orphan_pids = set(state["orphaned_peaks"].keys())
            state["orphaned_sack_owner"] = {
                s: p for s, p in state["orphaned_sack_owner"].items()
                if p in active_orphan_pids
            }

        # ── 10. Advance state machine for carried sacks ───────
        door_cx_sm = session.door.centroid[0]
        approach_side_sm = cfg.get("door_approach_side", "right")
        if approach_side_sm == "right":
            approach_open = door_cx_sm + cfg["peak_window_px"]
        else:
            approach_open = door_cx_sm - cfg["peak_window_px"]
        for sid, owner_pid in sack_owner_clean.items():
            rec = session.sack_sm.get(sid)
            if rec is None:
                continue
            if rec.state in (SackState.CONFIRMED, SackState.PICKED_UP):
                rec.transition(SackState.CARRIED, fn)
            if rec.state == SackState.CARRIED:
                pid_cx = state["person_prev_cx"].get(owner_pid, 0)
                if pid_cx <= approach_open:
                    rec.transition(SackState.APPROACHING, fn)

        # ── 11. Draw ──────────────────────────────────────────
        _elapsed = int(now - t_start)
        _draw_frame(
            frame, session, box_owner, fps, fn,
            frame_stats={
                "raw":       len(raw_sacks),
                "boxes":     len(raw_boxes),
                "still":     assign_stats.get("still", 0),
                "assigned":  assign_stats.get("assigned", 0),
                "crossings": n_crossings,
                "ghosts":    len(list(session.ghost_sacks.iter_ghosts())),
            },
            warning=ui_warning,
            elapsed=f"{_elapsed // 60}:{_elapsed % 60:02d}",
            src_fps=src_fps,
        )

        if save_output and out:
            out.write(frame)

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

    log_path = f"delivery_log_{VERSION_TAG}.json"
    with open(log_path, "w") as f:
        json.dump({
            "total_sacks":          state["total_sacks_counted"],
            "total_boxes":          session.box_pipeline.total_counted,
            "reid_events":          session.relinker.reid_events,
            "deliveries":           state["delivery_log"] + session.box_pipeline.delivery_log,
            "anomalies":            state["anomaly_log"],
            "person_peak_delivery": state["person_peak_delivery"],
            "analytics":            analytics_summary,
        }, f, indent=2)
    print(f"  Delivery log saved -> {log_path}\n")


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
        candidates=state.get("door_candidates"),
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
