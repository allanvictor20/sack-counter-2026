"""
sack_counter.exit — Exit sack counting module.

Public surface
--------------
run_exit_counter   : Main entry point (called from run.py in exit mode).
ExitPipelineState  : State container (useful for testing).
LandingZone        : Landing zone polygon geometry.
LandingZoneTracker : Motion tracker for landing zone sacks.
"""

from .exit_main import run_exit_counter
from .exit_state import ExitPipelineState
from .landing_zone import LandingZone, calibrate_landing_zone
from .exit_landing import LandingZoneTracker

__all__ = [
    "run_exit_counter",
    "ExitPipelineState",
    "LandingZone",
    "calibrate_landing_zone",
    "LandingZoneTracker",
]
