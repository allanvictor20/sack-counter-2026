"""
analytics.py — Analytics layer for Sack Counter v20.

Transforms raw delivery events into an operations-intelligence report:

    * Deliveries per worker
    * Average load size per carrier
    * Peak throughput (deliveries / minute)
    * Hourly-style trends (by N-frame bucket)
    * Ownership confidence statistics
    * ReID event summary

The :class:`AnalyticsEngine` is fed events incrementally during the main
loop and produces a final summary dict on demand.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkerStats:
    """
    Accumulated per-worker delivery statistics.

    Attributes:
        worker_id:      Canonical person ID.
        sack_deliveries: List of peak counts per delivery pass.
        box_deliveries:  List of box counts per delivery pass.
        confidence_scores: Ownership confidence at each delivery.
        first_delivery_frame: Frame number of first delivery.
        last_delivery_frame:  Frame number of most recent delivery.
    """
    worker_id:             int
    sack_deliveries:       list[int]   = field(default_factory=list)
    box_deliveries:        list[int]   = field(default_factory=list)
    confidence_scores:     list[float] = field(default_factory=list)
    first_delivery_frame:  int         = 0
    last_delivery_frame:   int         = 0

    # ── Derived properties ────────────────────────────────────────

    @property
    def total_sacks(self) -> int:
        """Total sacks delivered across all passes."""
        return sum(self.sack_deliveries)

    @property
    def total_boxes(self) -> int:
        """Total boxes delivered across all passes."""
        return sum(self.box_deliveries)

    @property
    def total_items(self) -> int:
        """Combined sack + box deliveries."""
        return self.total_sacks + self.total_boxes

    @property
    def avg_load_size(self) -> float:
        """Mean items per delivery pass (sacks only)."""
        if not self.sack_deliveries:
            return 0.0
        return self.total_sacks / len(self.sack_deliveries)

    @property
    def avg_confidence(self) -> float:
        """Mean ownership confidence across all deliveries."""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)

    @property
    def num_passes(self) -> int:
        """Number of complete delivery passes recorded."""
        return len(self.sack_deliveries) + len(self.box_deliveries)


class AnalyticsEngine:
    """
    Incremental analytics accumulator for the Sack Counter pipeline.

    Feed delivery events via :meth:`record_sack_delivery` and
    :meth:`record_box_delivery` during the main loop. Call
    :meth:`summary` at the end to get the full report dict.

    Args:
        src_fps: Video source frame rate, used for time-based metrics.
    """

    def __init__(self, src_fps: float = 20.0) -> None:
        self._src_fps     = max(src_fps, 1.0)
        self._workers:    dict[int, WorkerStats] = {}
        self._throughput: list[tuple[int, int]]  = []   # (frame, count)
        self._bucket_size = int(self._src_fps * 60)      # 1-minute buckets
        self._bucket_counts: dict[int, int]      = defaultdict(int)
        self._anomaly_count   = 0
        self._reid_count      = 0
        self._start_time      = time.time()
        self._total_frames    = 0

    # ── Feed methods ──────────────────────────────────────────────

    def record_sack_delivery(self, worker_id: int, peak_count: int,
                              confidence: float, frame_no: int) -> None:
        """
        Record a sack delivery event.

        Args:
            worker_id:   Canonical person ID.
            peak_count:  Peak sack count for this pass.
            confidence:  Ownership confidence score.
            frame_no:    Current frame number.
        """
        ws = self._get_or_create(worker_id, frame_no)
        ws.sack_deliveries.append(peak_count)
        ws.confidence_scores.append(confidence)
        ws.last_delivery_frame = frame_no

        self._throughput.append((frame_no, peak_count))
        bucket = frame_no // self._bucket_size
        self._bucket_counts[bucket] += peak_count

    def record_box_delivery(self, worker_id: int, box_count: int,
                             frame_no: int) -> None:
        """
        Record a box delivery event.

        Args:
            worker_id:  Canonical person ID.
            box_count:  Number of boxes in this pass.
            frame_no:   Current frame number.
        """
        ws = self._get_or_create(worker_id, frame_no)
        ws.box_deliveries.append(box_count)
        ws.last_delivery_frame = frame_no

        bucket = frame_no // self._bucket_size
        self._bucket_counts[bucket] += box_count

    def record_anomaly(self) -> None:
        """Increment the low-confidence anomaly counter."""
        self._anomaly_count += 1

    def record_reid_event(self) -> None:
        """Increment the ReID re-link counter."""
        self._reid_count += 1

    def update_frame(self, frame_no: int) -> None:
        """Called once per frame to keep total-frame count in sync."""
        self._total_frames = frame_no

    # ── Summary ───────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """
        Generate a complete analytics summary.

        Returns:
            Dictionary with keys: workers, totals, throughput,
            trend_by_minute, confidence, pipeline.
        """
        workers_out = {}
        for wid, ws in self._workers.items():
            workers_out[wid] = {
                "total_sacks":    ws.total_sacks,
                "total_boxes":    ws.total_boxes,
                "total_items":    ws.total_items,
                "avg_load_size":  round(ws.avg_load_size, 2),
                "num_passes":     ws.num_passes,
                "avg_confidence": round(ws.avg_confidence, 3),
                "first_delivery_frame": ws.first_delivery_frame,
                "last_delivery_frame":  ws.last_delivery_frame,
            }

        grand_sacks = sum(ws.total_sacks for ws in self._workers.values())
        grand_boxes = sum(ws.total_boxes for ws in self._workers.values())
        elapsed_min = max(
            self._total_frames / (self._src_fps * 60.0), 1 / 60.0
        )
        peak_throughput = self._peak_throughput_per_minute()

        trend = {
            f"minute_{b+1}": cnt
            for b, cnt in sorted(self._bucket_counts.items())
        }

        all_conf = [c for ws in self._workers.values()
                    for c in ws.confidence_scores]
        conf_stats = {
            "mean":  round(sum(all_conf) / len(all_conf), 3) if all_conf else 0.0,
            "min":   round(min(all_conf), 3) if all_conf else 0.0,
            "max":   round(max(all_conf), 3) if all_conf else 0.0,
            "anomalies": self._anomaly_count,
        }

        return {
            "workers": workers_out,
            "totals": {
                "sacks":          grand_sacks,
                "boxes":          grand_boxes,
                "combined":       grand_sacks + grand_boxes,
                "active_workers": len(self._workers),
            },
            "throughput": {
                "items_per_minute": round((grand_sacks + grand_boxes) / elapsed_min, 2),
                "peak_per_minute":  peak_throughput,
                "elapsed_minutes":  round(elapsed_min, 2),
            },
            "trend_by_minute": trend,
            "confidence":      conf_stats,
            "pipeline": {
                "total_frames": self._total_frames,
                "reid_events":  self._reid_count,
            },
        }

    def worker_stats(self, worker_id: int) -> WorkerStats | None:
        """Return raw WorkerStats for *worker_id*, or None."""
        return self._workers.get(worker_id)

    # ── Private helpers ───────────────────────────────────────────

    def _get_or_create(self, worker_id: int, frame_no: int) -> WorkerStats:
        if worker_id not in self._workers:
            self._workers[worker_id] = WorkerStats(
                worker_id=worker_id,
                first_delivery_frame=frame_no,
            )
        return self._workers[worker_id]

    def _peak_throughput_per_minute(self) -> float:
        """Return the highest delivery rate in any single minute bucket."""
        if not self._bucket_counts:
            return 0.0
        return float(max(self._bucket_counts.values()))
