"""
pipeline/ — Core pipeline modules for Sack Counter v21.

Modules
-------
state_machine   : Per-sack lifecycle states (DETECTED → DELIVERED).
analytics       : Per-worker stats, throughput, minute-by-minute trends.
ground_memory   : Memory of sacks already resting on the floor.
orchestrator    : PipelineSession — owns all components, replaces loose run() vars.
person_tracker  : update_persons, timeout_persons.
sack_tracker    : update_sacks.
counting        : update_peak_counts.
door_crossing   : update_door_zone, process_door_disappearance, check_door_reentry.
reporter        : print_report.
"""

from .state_machine import SackState, SackRecord, SackStateMachineRegistry
from .analytics import AnalyticsEngine, WorkerStats
from .ground_memory import GroundMemory, GroundObject
from .orchestrator import PipelineSession
from .state import PipelineState

__all__ = [
    "SackState", "SackRecord", "SackStateMachineRegistry",
    "AnalyticsEngine", "WorkerStats",
    "GroundMemory", "GroundObject",
    "PipelineSession",
    "PipelineState",
]
