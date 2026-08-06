"""
exit_reporter.py — End-of-session reporting for the exit sack counter.

Prints a formatted summary to stdout and writes a JSON log file.
"""

from __future__ import annotations

import json
import datetime
from typing import TYPE_CHECKING

from ..session_log import session_log_path

if TYPE_CHECKING:
    from .exit_state import ExitPipelineState


def print_exit_report(
    state: "ExitPipelineState",
    source: str,
    total_frames: int,
    src_fps: float,
    use_door_crossing: bool = False,
) -> None:
    """
    Print a formatted end-of-session summary to stdout.

    Args:
        state:              ExitPipelineState.
        source:             Video source path or 'webcam'.
        total_frames:       Total frames processed.
        src_fps:            Source video FPS.
        use_door_crossing:  Whether door-crossing was active this run.
                            When False (the default), door_count is
                            always 0 by design — the report reflects
                            that honestly instead of implying a
                            discrepancy check that never ran.
    """
    from .exit_landing import reconcile_counts as _reconcile

    best_count = _reconcile(state)
    door_count = state["total_sacks_out"]
    land_count = state["landing_exit_count"]
    duration_s = total_frames / max(src_fps, 1.0)

    sep = "=" * 60
    print(f"\n{sep}")
    print("  EXIT COUNTER — SESSION REPORT")
    print(f"  Source  : {source}")
    print(f"  Frames  : {total_frames}  ({duration_s:.1f} s @ {src_fps:.1f} fps)")
    print(sep)
    if use_door_crossing:
        print(f"  Sacks exited (door crossing) : {door_count}")
        print(f"  Sacks exited (landing zone)  : {land_count}")
        print(f"  Best estimate (reconciled)   : {best_count}")
    else:
        print(f"  Sacks exited (landing zone)  : {land_count}")
        print("  (door-crossing disabled — landing zone is the sole counter;")
        print("   pass --use-door-crossing to enable it as a cross-check)")
    print(sep)

    if use_door_crossing:
        if state["discrepancy_flags"]:
            print(f"  ⚠  {len(state['discrepancy_flags'])} discrepancy event(s) detected")
            for flag in state["discrepancy_flags"][-3:]:  # show last 3
                print(
                    f"     frame={flag['frame']}  "
                    f"door={flag['door_count']}  "
                    f"landing={flag['landing_count']}  "
                    f"diff={flag['diff']}"
                )
        else:
            print("  ✓  Door count and landing count agree")
        print(sep)

    print(f"  Exit events in log : {len(state['exit_log'])}")
    print(f"{sep}\n")


def save_exit_log(
    state: "ExitPipelineState",
    source: str,
    total_frames: int,
    src_fps: float,
    output_path: str | None = None,
) -> str:
    """
    Write the full exit log to a JSON file.

    Args:
        state:        ExitPipelineState.
        source:       Video source path or 'webcam'.
        total_frames: Total frames processed.
        src_fps:      Source video FPS.
        output_path:  File path to write.  Defaults to
                      ``exit_log_<timestamp>.json`` in the current directory.

    Returns:
        Path of the file that was written.
    """
    from .exit_landing import reconcile_counts as _reconcile

    if output_path is None:
        output_path = session_log_path("exit_log")

    payload = {
        "source":             source,
        "total_frames":       total_frames,
        "src_fps":            src_fps,
        "duration_seconds":   total_frames / max(src_fps, 1.0),
        "total_sacks_out":    state["total_sacks_out"],
        "landing_exit_count": state["landing_exit_count"],
        "reconciled_count":   _reconcile(state),
        "exit_log":           state["exit_log"],
        "discrepancy_flags":  state["discrepancy_flags"],
        "generated_at":       datetime.datetime.now().isoformat(),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"  Exit log saved -> {output_path}")
    return output_path
