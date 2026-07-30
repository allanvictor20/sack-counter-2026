"""
test_exit_reporter.py — Unit tests for exit_reporter.py.

Focused on print_exit_report's use_door_crossing-aware behavior, added
when door-crossing was demoted from primary to optional counting signal.
With door-crossing disabled (the new default), total_sacks_out stays 0
by design — the report must reflect that honestly rather than printing
a misleading "door count: 0" or falsely claiming door/landing counts
"agree" when door-crossing never ran at all.
"""

import io
import contextlib

import pytest

from sack_counter.exit.exit_state import ExitPipelineState
from sack_counter.exit.exit_reporter import print_exit_report


def _run_report(state, use_door_crossing):
    """Capture stdout from print_exit_report for assertions."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_exit_report(
            state, source="test.mp4", total_frames=100, src_fps=20.0,
            use_door_crossing=use_door_crossing,
        )
    return buf.getvalue()


class TestPrintExitReportDoorCrossingDisabled:
    """Default behavior: use_door_crossing=False."""

    def test_does_not_print_door_crossing_line(self, state):
        state["landing_exit_count"] = 5
        output = _run_report(state, use_door_crossing=False)
        assert "door crossing" not in output.lower()

    def test_prints_landing_zone_count(self, state):
        state["landing_exit_count"] = 5
        output = _run_report(state, use_door_crossing=False)
        assert "Sacks exited (landing zone)  : 5" in output

    def test_explains_door_crossing_is_disabled(self, state):
        output = _run_report(state, use_door_crossing=False)
        assert "door-crossing disabled" in output

    def test_does_not_claim_counts_agree(self, state):
        """Critical: must not print the misleading 'door count and
        landing count agree' message when door-crossing never ran."""
        state["landing_exit_count"] = 5
        output = _run_report(state, use_door_crossing=False)
        assert "agree" not in output.lower()

    def test_does_not_print_discrepancy_section(self, state):
        state["discrepancy_flags"].append({
            "frame": 10, "door_count": 0, "landing_count": 5, "diff": 5,
        })
        output = _run_report(state, use_door_crossing=False)
        # Even if a stale discrepancy flag exists, it must not be shown
        # when use_door_crossing is False for this report call.
        assert "discrepancy event" not in output.lower()


class TestPrintExitReportDoorCrossingEnabled:
    """Behavior when --use-door-crossing is explicitly enabled."""

    def test_prints_both_counts(self, state):
        state["total_sacks_out"]    = 4
        state["landing_exit_count"] = 5
        output = _run_report(state, use_door_crossing=True)
        assert "Sacks exited (door crossing) : 4" in output
        assert "Sacks exited (landing zone)  : 5" in output

    def test_prints_best_estimate(self, state):
        state["total_sacks_out"]    = 4
        state["landing_exit_count"] = 5
        output = _run_report(state, use_door_crossing=True)
        assert "Best estimate (reconciled)   : 5" in output

    def test_prints_agree_message_when_no_discrepancy(self, state):
        state["total_sacks_out"]    = 5
        state["landing_exit_count"] = 5
        output = _run_report(state, use_door_crossing=True)
        assert "agree" in output.lower()

    def test_prints_discrepancy_warning_when_flagged(self, state):
        state["discrepancy_flags"].append({
            "frame": 10, "door_count": 2, "landing_count": 5, "diff": 3,
        })
        output = _run_report(state, use_door_crossing=True)
        assert "discrepancy event" in output.lower()


class TestPrintExitReportDoesNotCrash:

    def test_empty_state_door_disabled(self, state):
        output = _run_report(state, use_door_crossing=False)
        assert "EXIT COUNTER" in output

    def test_empty_state_door_enabled(self, state):
        output = _run_report(state, use_door_crossing=True)
        assert "EXIT COUNTER" in output

    def test_default_use_door_crossing_is_false(self, state):
        """Calling without explicit use_door_crossing should default to
        the disabled/landing-zone-only report format."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_exit_report(state, source="test.mp4",
                             total_frames=100, src_fps=20.0)
        output = buf.getvalue()
        assert "door-crossing disabled" in output
