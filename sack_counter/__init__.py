"""
Sack-Per-Person Counter v21 — modular package.

New in v20
----------
* ``PipelineState._resolve()`` removed — dead code never called internally.
* ``AssignBugState`` renamed to ``CarrierGuardState`` — these are first-class
  carrier management fields, not ad-hoc patches.
* ``assign_sacks_hungarian`` now accepts ``persons_past_gate`` and excludes
  those IDs from the Hungarian cost matrix (P#457-class bystander fix).
* ``GroundMemory.cleanup()`` prunes stale grounded records via TTL to prevent
  unbounded memory growth on long sessions.
* ``GroundMemory._cosine`` local helper removed; ``DeepEmbedder.cosine`` used
  directly everywhere to eliminate a duplicate implementation.
* All version strings updated from v18/v19 → v20.

New in v18-v19 (retained)
--------------------------
* pipeline/ sub-package:
    - SackStateMachineRegistry  — per-sack lifecycle states
    - AnalyticsEngine           — per-worker stats, throughput trends
    - GroundMemory              — memory of floor sacks
    - PipelineSession           — session container replacing loose run() vars
* YAML config support (config.yaml auto-discovered)
* Structured logging with file + console levels
* Type hints and docstrings throughout

Heavy imports (torch, cv2, ultralytics) are done lazily so that
lightweight modules (config, gate, confidence, trackers, state_machine,
analytics, ground_memory, pipeline.state) can be imported in tests and
utility scripts without requiring a GPU environment.
"""

# ── Always-safe imports (no torch / cv2 / ultralytics) ───────────────────────
from .config import DEFAULT_CFG, load_config
from .door_polygon import DoorPolygon, calibrate_door_polygon
from .confidence import ConfidenceTracker, confidence_class
from .pipeline import (
    PipelineSession,
    PipelineState,
    SackState,
    SackRecord,
    SackStateMachineRegistry,
    AnalyticsEngine,
    WorkerStats,
    GroundMemory,
    GroundObject,
)

__all__ = [
    "DEFAULT_CFG", "load_config",
    "DoorPolygon", "calibrate_door_polygon",
    "ConfidenceTracker", "confidence_class",
    "PipelineSession",
    "PipelineState",
    "SackState", "SackRecord", "SackStateMachineRegistry",
    "AnalyticsEngine", "WorkerStats",
    "GroundMemory", "GroundObject",
]


def _lazy_heavy():
    """
    Import heavy runtime dependencies (torch, cv2, ultralytics).

    Called automatically when run() or draw_* helpers are first used.
    This keeps test-suite imports fast and GPU-free.
    """
    global DeepEmbedder, SackMotionTracker, IdentityRelinker, OwnershipMemory
    global GhostSacks, assign_sacks_hungarian, association_score
    global carry_zone_bounds, BoxCountingPipeline
    global draw_person_box, draw_box_detections, draw_sack_boxes, draw_hud
    global run

    from .embedder import DeepEmbedder
    from .trackers import (
        SackMotionTracker, IdentityRelinker, OwnershipMemory, GhostSacks,
    )
    from .assignment import (
        assign_sacks_hungarian, association_score, carry_zone_bounds,
    )
    from .box_pipeline import BoxCountingPipeline
    from .drawing import (
        draw_person_box, draw_box_detections, draw_sack_boxes, draw_hud,
    )
    from .main import run

    __all__.extend([
        "DeepEmbedder",
        "SackMotionTracker", "IdentityRelinker", "OwnershipMemory", "GhostSacks",
        "assign_sacks_hungarian", "association_score", "carry_zone_bounds",
        "BoxCountingPipeline",
        "draw_person_box", "draw_box_detections", "draw_sack_boxes",
        "draw_hud",
        "run",
    ])


def __getattr__(name: str):
    """Lazy-load heavy symbols on first access."""
    _lazy_heavy()
    import sys
    mod = sys.modules[__name__]
    if hasattr(mod, name):
        return getattr(mod, name)
    raise AttributeError(f"module 'sack_counter' has no attribute {name!r}")
