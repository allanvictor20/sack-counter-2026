"""
detections.py — Turn a tracker result into the pipeline's tuples.

Split out of the frame loop so it can be tested without a model: it is a
pure function over whatever the detector returned, and it is where the
per-class confidence floors are actually applied.

The tuple shapes are the pipeline's own and are load-bearing — sacks
carry their centroid because assignment and crossing both need it, while
persons and boxes do not.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class FrameDetections(NamedTuple):
    """One frame's detections, split by class."""

    persons: list[tuple]   # (tid, x1, y1, x2, y2, score)
    sacks:   list[tuple]   # (tid, x1, y1, x2, y2, cx, cy, score)
    boxes:   list[tuple]   # (tid, x1, y1, x2, y2, score)


def parse_detections(results: Any, cls_idx: dict[str, int],
                     cfg: dict) -> FrameDetections:
    """
    Split a tracker result into per-class detection tuples.

    Detections without a track id are dropped: everything downstream is
    keyed on a stable id, so an untracked box has nothing to attach to.

    Args:
        results: What ``model.track()`` returned.
        cls_idx: ``{"person": i, "sack": j, "box": k}`` from
                 :func:`~sack_counter.model_classes.resolve_class_indices`.
        cfg:     Config, read for the per-class confidence floors.

    Returns:
        A :class:`FrameDetections` triple.  Empty lists when the frame
        held nothing, or when the detector returned nothing at all.
    """
    persons: list[tuple] = []
    sacks:   list[tuple] = []
    boxes:   list[tuple] = []

    if not results or results[0].boxes is None:
        return FrameDetections(persons, sacks, boxes)

    cls_person = cls_idx["person"]
    cls_sack   = cls_idx["sack"]
    cls_box    = cls_idx["box"]
    min_person = cfg["conf_person"]
    min_sack   = cfg["conf_sack"]
    min_box    = cfg["conf_box"]

    for det in results[0].boxes:
        if det.id is None:
            continue
        cls   = int(det.cls[0])
        score = float(det.conf[0])
        tid   = int(det.id[0])
        x1, y1, x2, y2 = map(int, det.xyxy[0])

        if cls == cls_person and score >= min_person:
            persons.append((tid, x1, y1, x2, y2, score))
        elif cls == cls_sack and score >= min_sack:
            sacks.append((tid, x1, y1, x2, y2,
                          (x1 + x2) // 2, (y1 + y2) // 2, score))
        elif cls == cls_box and score >= min_box:
            boxes.append((tid, x1, y1, x2, y2, score))

    return FrameDetections(persons, sacks, boxes)


def detection_conf_floor(cfg: dict) -> float:
    """
    The single threshold to run the detector at.

    The tracker takes one confidence, but the pipeline wants a different
    floor per class, so it must run at the lowest of the three and filter
    afterwards — running higher would discard detections a class-specific
    floor would have kept.
    """
    return min(cfg["conf_sack"], cfg["conf_person"], cfg["conf_box"])
