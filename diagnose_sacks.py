"""
diagnose_sacks.py — Raw YOLO sack detection diagnostic.

Runs ONLY the YOLO model on your video — no door crossing logic, no
landing zone logic, no confirmation/hit-miss tracking, no GroundMemory,
nothing from the exit or entry pipeline. This shows EXACTLY what the
model itself sees, frame by frame, so we can tell whether a detection
problem is:

  (a) the model genuinely not seeing sacks on the ground (a model/data
      problem — would need fine-tuning), or
  (b) the model seeing them fine, but pipeline logic on top suppressing
      or losing them (a code problem — fixable without touching the model)

What it does
------------
1. Loads your existing best.pt model.
2. Runs raw detection (NOT tracking — just YOLO.predict) on every frame,
   at a LOW confidence floor (0.05) so we see everything the model
   considers even remotely possible, sack class only.
3. Prints a per-frame summary: how many sack detections, and their
   confidence scores.
4. Saves annotated frames (every Nth frame, configurable) to
   ./diagnostic_frames/ so you can visually inspect exactly what boxes
   the model drew and at what confidence, with NO other logic applied.
5. Prints an overall histogram of confidence scores across the whole
   video at the end, so we can see where most detections cluster.

Usage
-----
    python diagnose_sacks.py your_video.mp4

Optional:
    python diagnose_sacks.py your_video.mp4 --every 30 --conf-floor 0.05

Output
------
- Console: per-frame detection counts + confidence scores
- ./diagnostic_frames/frame_XXXXX.jpg : annotated frames at intervals
- Final summary: confidence histogram, frames with zero detections, etc.
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

from sack_counter import theme as T
from sack_counter.drawing import draw_count_block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Path to video file")
    ap.add_argument("--model", default="best.pt", help="Path to YOLO model")
    ap.add_argument("--every", type=int, default=30,
                    help="Save an annotated frame every N frames (default 30)")
    ap.add_argument("--conf-floor", type=float, default=0.05,
                    help="Minimum confidence to even report (default 0.05 — very low, "
                         "shows everything the model considers)")
    args = ap.parse_args()

    from ultralytics import YOLO

    print(f"\n{'='*70}")
    print(f"  RAW SACK DETECTION DIAGNOSTIC")
    print(f"  Video       : {args.video}")
    print(f"  Model       : {args.model}")
    print(f"  Conf floor  : {args.conf_floor}  (very low — shows everything)")
    print(f"{'='*70}\n")

    model = YOLO(args.model)

    # Find sack class index
    sack_cls = None
    for idx, name in model.names.items():
        if name.lower() == "sack":
            sack_cls = int(idx)
            break
    if sack_cls is None:
        print(f"ERROR: No 'sack' class in model. Classes: {model.names}")
        sys.exit(1)
    print(f"  Sack class index: {sack_cls}\n")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.video}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Resolution  : {fw}x{fh}")
    print(f"  Total frames: {total_frames}  ({total_frames/fps:.1f}s @ {fps:.1f}fps)\n")

    out_dir = Path("diagnostic_frames")
    out_dir.mkdir(exist_ok=True)

    all_confidences = []
    zero_detection_frames = []
    fn = 0

    print(f"  {'Frame':>6} | {'#Sacks':>6} | Confidence scores")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*50}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fn += 1

        results = model.predict(
            frame,
            conf=args.conf_floor,
            classes=[sack_cls],
            verbose=False,
        )

        confs = []
        boxes_xyxy = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            b = results[0].boxes
            confs = b.conf.cpu().numpy().tolist()
            boxes_xyxy = b.xyxy.cpu().numpy().astype(int).tolist()

        all_confidences.extend(confs)

        if len(confs) == 0:
            zero_detection_frames.append(fn)

        # Print every frame's summary (or throttle if video is long)
        if total_frames <= 300 or fn % 5 == 0 or len(confs) == 0:
            conf_str = ", ".join(f"{c:.2f}" for c in sorted(confs, reverse=True)[:8])
            if len(confs) > 8:
                conf_str += f", ... (+{len(confs)-8} more)"
            print(f"  {fn:>6} | {len(confs):>6} | {conf_str}")

        # Save annotated frame at intervals
        if fn % args.every == 0 or fn == 1:
            annotated = frame.copy()
            u = T.unit(annotated)
            # Same reading as the design's detection-check histogram:
            # accent for scores that survive the cut-off, neutrals below.
            for (x1, y1, x2, y2), c in zip(boxes_xyxy, confs):
                color = T.ACCENT if c >= 0.35 else \
                        T.ACCENT_400 if c >= 0.20 else T.NEUTRAL_400
                T.tracked_box(annotated, x1, y1, x2, y2, u, color,
                              f"{c:.2f}", label_ink=T.contrast_ink(color),
                              thickness=2.5, size=10.5)
            draw_count_block(
                annotated, u, "Sacks seen", len(confs),
                stats=(("Frame", fn), ("Cut-off", f"{args.conf_floor:.2f}")),
            )
            out_path = out_dir / f"frame_{fn:05d}.jpg"
            cv2.imwrite(str(out_path), annotated)

    cap.release()

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Total frames processed     : {fn}")
    print(f"  Frames with ZERO detections: {len(zero_detection_frames)}")
    if zero_detection_frames:
        sample = zero_detection_frames[:10]
        print(f"    (first few: {sample}{'...' if len(zero_detection_frames) > 10 else ''})")
    print(f"  Total sack detections      : {len(all_confidences)}")

    if all_confidences:
        arr = np.array(all_confidences)
        print(f"\n  Confidence score distribution:")
        print(f"    min    : {arr.min():.3f}")
        print(f"    max    : {arr.max():.3f}")
        print(f"    mean   : {arr.mean():.3f}")
        print(f"    median : {np.median(arr):.3f}")

        buckets = [
            ("0.05-0.15", (arr >= 0.05) & (arr < 0.15)),
            ("0.15-0.25", (arr >= 0.15) & (arr < 0.25)),
            ("0.25-0.35", (arr >= 0.25) & (arr < 0.35)),
            ("0.35-0.50", (arr >= 0.35) & (arr < 0.50)),
            ("0.50-0.70", (arr >= 0.50) & (arr < 0.70)),
            ("0.70-1.00", (arr >= 0.70) & (arr <= 1.00)),
        ]
        print(f"\n  Histogram:")
        for label, mask in buckets:
            n = int(mask.sum())
            bar = "#" * min(60, n // max(1, len(all_confidences)//200 or 1))
            print(f"    {label} : {n:>5}  {bar}")
    else:
        print("\n  !! NO SACK DETECTIONS AT ALL, EVEN AT conf=0.05 !!")
        print("  This points to a MODEL problem, not a pipeline/code problem.")

    print(f"\n  Annotated sample frames saved to: {out_dir.resolve()}")
    print(f"  -> Open a few of these and visually check if sacks ARE boxed")
    print(f"     (even faintly/low-confidence) or NOT boxed at all.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
