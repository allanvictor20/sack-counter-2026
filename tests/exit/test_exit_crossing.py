"""
test_exit_crossing.py — Unit tests for door crossing detection logic.

Uses FakeDoorPolygon where:
  centroid = (320, 240)
  door strip (contains() returns True for): x in [310, 330], y in [100, 380]
  normal_vec = (-1, 0)  →  project(x, y) = 320 - x

  x > 320  →  negative projection  →  CORRIDOR side
  x < 320  →  positive projection  →  ROOM side
  x = 320  →  zero (at door)

REDESIGNED trigger (see exit_crossing.py module docstring for the full
rationale): a sack is registered as a tentative crossing the FIRST time
it is confirmed AND its position is inside the door polygon OR within
`door_proximity_px` of it (measured via |projection| <= proximity).
There is no "room → corridor transition" anymore, because the room
interior is never actually visible to the camera — a sack's first-ever
detection IS the crossing signal, not a later frame compared to an
earlier one.
"""

import pytest
from sack_counter.exit.exit_crossing import (
    update_exit_crossings,
    confirm_tentative_crossings,
    expire_old_projections,
)


class TestUpdateExitCrossings:

    def _add_confirmed(self, state, sid, cx, cy=240):
        """Helper: add a confirmed sack at (cx, cy), with NO prior
        projection recorded — simulating its first-ever sighting."""
        state["confirmed_sacks"].add(sid)
        state["sack_positions"][sid] = (cx, cy)

    def test_first_sighting_far_from_door_no_event(self, state, door):
        """A sack whose first-ever sighting is far from the door entirely
        (not inside it, not within proximity) should NOT trigger a
        tentative crossing — that's the landing zone's job instead."""
        self._add_confirmed(state, sid=1, cx=175, cy=375)  # landing zone area, far from door at x=320

        events = update_exit_crossings(state, door, fn=10, door_proximity_px=60)
        assert events == []
        assert 1 not in state["tentative_crossings"]

    def test_first_sighting_inside_door_polygon_triggers(self, state, door):
        """A sack whose first-ever sighting is INSIDE the door polygon
        itself (x in [310,330]) should trigger a tentative crossing."""
        sid = 1
        self._add_confirmed(state, sid=sid, cx=320, cy=240)  # dead center of door

        events = update_exit_crossings(state, door, fn=5)
        assert len(events) == 1
        assert events[0]["type"] == "tentative"
        assert events[0]["sack_id"] == sid
        assert events[0]["trigger"] == "first_seen_at_door"
        assert sid in state["tentative_crossings"]

    def test_first_sighting_near_door_within_proximity_triggers(self, state, door):
        """A sack first seen just outside the door polygon itself, but
        within door_proximity_px of the door centroid's projection,
        should still trigger — covers the case of a sack landing just
        past the door frame, not literally inside the doorway pixels."""
        sid = 2
        # x=370: proj = 320-370 = -50, abs(50) <= proximity(60) -> should trigger
        self._add_confirmed(state, sid=sid, cx=370, cy=240)

        events = update_exit_crossings(state, door, fn=5, door_proximity_px=60)
        assert len(events) == 1
        assert sid in state["tentative_crossings"]

    def test_first_sighting_beyond_proximity_does_not_trigger(self, state, door):
        """A sack first seen well beyond the proximity margin should NOT
        trigger a door-crossing tentative — it's outside the door's
        detection radius entirely."""
        sid = 3
        # x=450: proj = 320-450 = -130, abs(130) > proximity(60) -> no trigger
        self._add_confirmed(state, sid=sid, cx=450, cy=240)

        events = update_exit_crossings(state, door, fn=5, door_proximity_px=60)
        assert events == []
        assert sid not in state["tentative_crossings"]

    def test_second_sighting_of_same_sack_does_not_retrigger(self, state, door):
        """Once a sack has a recorded prev_proj (i.e. has been seen
        before), a later frame must NOT trigger a new tentative crossing
        even if it's near the door — only the sack's first-ever sighting
        can trigger this."""
        sid = 4
        self._add_confirmed(state, sid=sid, cx=320, cy=240)  # first sighting, at door

        first_events = update_exit_crossings(state, door, fn=5)
        assert len(first_events) == 1

        # Same sack, later frame — should not re-trigger
        state["sack_positions"][sid] = (322, 240)  # still at door
        second_events = update_exit_crossings(state, door, fn=6)
        assert second_events == []

    def test_already_crossed_sack_not_recounted(self, state, door):
        """Sack already in crossed_sacks is ignored even on first call."""
        sid = 1
        self._add_confirmed(state, sid=sid, cx=320, cy=240)
        state["crossed_sacks"].add(sid)

        events = update_exit_crossings(state, door, fn=5)
        assert events == []

    def test_already_tentative_not_re_added(self, state, door):
        """Sack already in tentative_crossings doesn't get a second entry,
        even if its position changes (already-pending tentatives are
        skipped before the first-sighting check even runs)."""
        sid = 1
        self._add_confirmed(state, sid=sid, cx=320, cy=240)
        state["tentative_crossings"][sid] = 3

        events = update_exit_crossings(state, door, fn=5)
        assert events == []
        assert state["tentative_crossings"][sid] == 3  # unchanged

    def test_no_event_for_empty_confirmed_set(self, state, door):
        events = update_exit_crossings(state, door, fn=1)
        assert events == []

    def test_sack_with_no_position_skipped_safely(self, state, door):
        """A confirmed sack with no recorded position must not crash —
        it's simply skipped this frame."""
        state["confirmed_sacks"].add(99)
        # Deliberately no sack_positions[99] entry
        events = update_exit_crossings(state, door, fn=5)
        assert events == []


class TestConfirmTentativeCrossings:

    def test_tentative_confirmed_when_landed(self, state):
        sid = 1
        state["tentative_crossings"][sid] = 5
        state["landed_sacks"].add(sid)

        confirmed, cancelled = confirm_tentative_crossings(state, fn=10, confirm_timeout=45)
        assert len(confirmed) == 1
        assert confirmed[0]["sack_id"] == sid
        assert confirmed[0]["type"] == "confirmed"
        assert sid in state["crossed_sacks"]
        assert state["total_sacks_out"] == 1
        assert sid not in state["tentative_crossings"]

    def test_tentative_cancelled_on_timeout(self, state):
        sid = 2
        state["tentative_crossings"][sid] = 1

        confirmed, cancelled = confirm_tentative_crossings(state, fn=50, confirm_timeout=45)
        assert len(cancelled) == 1
        assert cancelled[0]["sack_id"] == sid
        assert cancelled[0]["type"] == "cancelled"
        assert sid not in state["tentative_crossings"]
        assert state["total_sacks_out"] == 0

    def test_tentative_waits_within_timeout(self, state):
        sid = 3
        state["tentative_crossings"][sid] = 10

        confirmed, cancelled = confirm_tentative_crossings(state, fn=30, confirm_timeout=45)
        assert len(confirmed) == 0
        assert len(cancelled) == 0
        assert sid in state["tentative_crossings"]

    def test_multiple_tentatives_mixed_outcomes(self, state):
        state["tentative_crossings"][1] = 5
        state["tentative_crossings"][2] = 5
        state["landed_sacks"].add(1)   # sid 1 landed → confirm
        # sid 2: no landing, no timeout yet → pending

        confirmed, cancelled = confirm_tentative_crossings(state, fn=20, confirm_timeout=45)
        assert len(confirmed) == 1
        assert confirmed[0]["sack_id"] == 1
        assert 2 in state["tentative_crossings"]

    def test_count_increments_correctly(self, state):
        state["tentative_crossings"][1] = 1
        state["tentative_crossings"][2] = 1
        state["landed_sacks"].update({1, 2})

        confirm_tentative_crossings(state, fn=5, confirm_timeout=45)
        assert state["total_sacks_out"] == 2

    def test_event_added_to_exit_log(self, state):
        state["tentative_crossings"][7] = 1
        state["landed_sacks"].add(7)
        confirm_tentative_crossings(state, fn=5, confirm_timeout=45)
        assert any(e["sack_id"] == 7 for e in state["exit_log"])


class TestExpireOldProjections:

    def test_stale_projections_removed(self, state):
        state["sack_prev_proj"][10] = 50.0
        state["sack_curr_proj"][10] = 30.0
        expire_old_projections(state, active_sack_ids={1, 2})
        assert 10 not in state["sack_prev_proj"]
        assert 10 not in state["sack_curr_proj"]

    def test_active_projections_kept(self, state):
        state["sack_prev_proj"][5] = 20.0
        expire_old_projections(state, active_sack_ids={5})
        assert 5 in state["sack_prev_proj"]

    def test_empty_active_set_clears_all(self, state):
        state["sack_prev_proj"].update({1: 10.0, 2: 20.0})
        expire_old_projections(state, active_sack_ids=set())
        assert len(state["sack_prev_proj"]) == 0
