"""
orchestrator.py — Pipeline orchestrator for Sack Counter v21.

Changes from v19
----------------
* ``PipelineSession.__init__`` now accepts a ``calibration`` dict
  (``{"line_x": int, "direction": str, "offset": int}``) produced by
  ``calibration.calibrate()``.  The old ``gate_x`` / ``gate_gap``
  positional overrides are kept as a fallback for headless / CLI use.
* ``GateCounter`` replaced with ``DoorPolygon``. Call ``session.set_door(door)``
  after calibration before processing any frames.
* Flat ``state`` dict replaced by typed ``PipelineState`` (state.py).
  All existing call-sites using ``state["key"]`` continue to work via
  the dict-like interface on ``PipelineState``.
* ``OwnershipMemory`` gains ``iter_by_owner()`` — internal ``_owner``
  is no longer accessed from outside the class.
* ``_frame_no`` added to initial state so ``final_summary()`` is safe
  even before the first frame.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..config import load_config
from ..door_polygon import DoorPolygon
from ..confidence import ConfidenceTracker
from .analytics import AnalyticsEngine
from .ground_memory import GroundMemory
from .state_machine import SackStateMachineRegistry
from .state import PipelineState

# Heavy imports deferred so lightweight modules can load without torch/cv2
def _import_heavy():
    """Import GPU/CV dependencies on first PipelineSession construction."""
    global setup_logger, DeepEmbedder, SackMotionTracker, IdentityRelinker
    global OwnershipMemory, GhostSacks, BoxCountingPipeline
    from ..logger import setup_logger
    from ..embedder import DeepEmbedder
    from ..trackers import (
        SackMotionTracker, IdentityRelinker, OwnershipMemory, GhostSacks,
    )
    from ..box_pipeline import BoxCountingPipeline


class PipelineSession:
    """
    Container for all stateful components in a single video session.

    Create one instance per video/camera source.  Pass the instance into
    the frame-level helpers in ``main.py`` instead of passing individual
    objects.

    Args:
        cfg:           Configuration dictionary (from :func:`load_config`).
        frame_width:   Width of source video in pixels.
        frame_height:  Height of source video in pixels.
        src_fps:       Source video frame rate.
    """

    def __init__(
        self,
        cfg:          dict[str, Any],
        frame_width:  int,
        frame_height: int,
        src_fps:      float,
    ) -> None:
        _import_heavy()
        self.cfg          = cfg
        self.frame_width  = frame_width
        self.frame_height = frame_height
        self.src_fps      = src_fps

        # ── Derived timing constants ───────────────────────────────
        self.confirm_frames   = max(2, int(src_fps * cfg["confirm_secs"]))
        self.miss_frames      = max(2, int(src_fps * cfg["miss_secs"]))
        self.occlusion_frames = max(1, int(src_fps * cfg["occlusion_secs"]))
        self.reid_mem_frames  = max(2, int(src_fps * cfg["reid_memory_secs"]))
        self.suppress_frames  = max(1, int(src_fps * cfg["still_suppress_secs"]))

        # ── Logger ────────────────────────────────────────────────
        self.logger: logging.Logger = setup_logger()

        # ── Core components ───────────────────────────────────────
        self.embedder = DeepEmbedder()

        self.motion_tracker = SackMotionTracker(
            speed_thresh=cfg["still_speed_thresh"],
            speed_window=cfg["still_speed_window"],
        )
        self.ghost_sacks   = GhostSacks(max_frames=self.occlusion_frames)
        self.ownership_mem = OwnershipMemory(cfg["ownership_switch_margin"])
        self.conf_tracker  = ConfidenceTracker(cfg)
        self.relinker      = IdentityRelinker(cfg, self.embedder)
        self.relinker.set_memory_frames(self.reid_mem_frames)

        # ── Door polygon (set after calibration via set_door()) ────
        # door is None until set_door() is called before the main loop.
        self.door: Optional[DoorPolygon] = None

        # ── Box pipeline ──────────────────────────────────────────
        box_ownership_mem    = OwnershipMemory(cfg["ownership_switch_margin"])
        box_conf_tracker     = ConfidenceTracker(cfg)
        self.box_ownership_mem = box_ownership_mem
        self.box_conf_tracker  = box_conf_tracker
        self.box_pipeline = BoxCountingPipeline(
            confirm_frames = self.confirm_frames,
            miss_frames    = self.miss_frames,
            ownership_mem  = box_ownership_mem,
            conf_tracker   = box_conf_tracker,
            cfg            = cfg,
        )

        # ── New v18-v20 modules ───────────────────────────────────
        self.analytics     = AnalyticsEngine(src_fps=src_fps)
        self.ground_memory = GroundMemory(
            pickup_stable_frames=cfg["lift_owner_stable_frames"],
        )
        self.sack_sm = SackStateMachineRegistry()

        # ── Typed state container (replaces flat dict) ────────────
        self.state = PipelineState()
        # _frame_no initialised so final_summary() is safe before frame 1
        self.state["_frame_no"] = 0

    # ── Convenience accessors ─────────────────────────────────────────────────

    def set_door(self, door: "DoorPolygon") -> None:
        """
        Assign the calibrated door polygon.
        Must be called before the first frame is processed.

        Args:
            door: Calibrated DoorPolygon instance.
        """
        self.door = door

    def record_delivery(
        self,
        worker_id:  int,
        peak_count: int,
        confidence: float,
        frame_no:   int,
    ) -> None:
        """
        Push a sack delivery event to the analytics engine.

        Args:
            worker_id:  Canonical person ID.
            peak_count: Peak sack count for this pass.
            confidence: Ownership confidence score.
            frame_no:   Current frame number.
        """
        self.analytics.record_sack_delivery(
            worker_id, peak_count, confidence, frame_no
        )

    def record_box_delivery(
        self,
        worker_id: int,
        box_count:  int,
        frame_no:   int,
    ) -> None:
        """
        Push a box delivery event to the analytics engine.

        Args:
            worker_id: Canonical person ID.
            box_count: Number of boxes in this pass.
            frame_no:  Current frame number.
        """
        self.analytics.record_box_delivery(worker_id, box_count, frame_no)

    def final_summary(self) -> dict[str, Any]:
        """
        Return the complete analytics summary for end-of-session reporting.

        Returns:
            Dict from :meth:`AnalyticsEngine.summary`.
        """
        self.analytics.update_frame(self.state["_frame_no"])
        return self.analytics.summary()
