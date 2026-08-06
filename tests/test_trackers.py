"""
test_trackers.py — The identity and ownership logic.

IdentityRelinker decides *who gets credit* for a load: a re-link that
fires wrongly merges two workers' totals, and one that fails to fire
splits one worker across two rows.  Nothing in the suite covered it.

These tests use a stub embedder so nothing here loads torch — the
relinker only ever calls ``DeepEmbedder.cosine``, which is a static dot
product over already-normalised vectors.
"""

import numpy as np
import pytest

from sack_counter.config import DEFAULT_CFG
from sack_counter.trackers import (
    GhostSacks, IdentityRelinker, OwnershipMemory, SackMotionTracker,
)


class _NullLogger:
    """The relinker logs re-links; tests do not care what it says."""

    def info(self, *args, **kwargs):
        pass

    debug = warning = error = info


def _emb(*values) -> np.ndarray:
    """A unit vector, so cosine() is a plain dot product."""
    vec = np.array(values, dtype=np.float32)
    return vec / (np.linalg.norm(vec) + 1e-9)


def _relinker(**overrides) -> IdentityRelinker:
    cfg = dict(DEFAULT_CFG)
    cfg.update(overrides)
    relinker = IdentityRelinker(cfg, embedder=None)
    relinker.set_memory_frames(30)
    return relinker


# ── SackMotionTracker ─────────────────────────────────────────

class TestSackMotionTracker:
    def test_needs_three_samples_before_judging(self):
        tracker = SackMotionTracker(speed_thresh=2.0, speed_window=8)
        tracker.update(1, 100, 100)
        tracker.update(1, 100, 100)
        # Two samples is one displacement — not enough to call it still.
        assert tracker.is_still(1) is False

    def test_stationary_sack_is_still(self):
        tracker = SackMotionTracker(speed_thresh=2.0, speed_window=8)
        for _ in range(5):
            tracker.update(7, 100, 100)
        assert tracker.is_still(7) is True

    def test_moving_sack_is_not_still(self):
        tracker = SackMotionTracker(speed_thresh=2.0, speed_window=8)
        for step in range(5):
            tracker.update(7, 100 + step * 20, 100)
        assert tracker.is_still(7) is False

    def test_unknown_sack_is_not_still(self):
        tracker = SackMotionTracker(speed_thresh=2.0, speed_window=8)
        assert tracker.is_still(999) is False

    def test_cleanup_drops_inactive_history(self):
        tracker = SackMotionTracker(speed_thresh=2.0, speed_window=8)
        for _ in range(4):
            tracker.update(1, 10, 10)
            tracker.update(2, 20, 20)
        tracker.cleanup({1})
        assert 1 in tracker.history
        assert 2 not in tracker.history


# ── OwnershipMemory ───────────────────────────────────────────

class TestOwnershipMemory:
    def test_first_claim_wins_uncontested(self):
        mem = OwnershipMemory(switch_margin=0.15)
        assert mem.update(sid=1, new_pid=7, new_score=0.4) == 7
        assert mem.get(1) == 7

    def test_incumbent_holds_against_a_marginal_challenger(self):
        mem = OwnershipMemory(switch_margin=0.15)
        mem.update(1, 7, 0.60)
        # 0.70 beats 0.60 but not by the margin — hysteresis holds.
        assert mem.update(1, 9, 0.70) == 7
        assert mem.get(1) == 7

    def test_challenger_wins_by_the_margin(self):
        mem = OwnershipMemory(switch_margin=0.15)
        mem.update(1, 7, 0.60)
        assert mem.update(1, 9, 0.75) == 9
        assert mem.get(1) == 9

    def test_incumbent_score_is_refreshed_not_frozen(self):
        """
        A held owner's score must track downward, or the incumbent
        becomes progressively harder to unseat as its stale high score
        keeps raising the bar.
        """
        mem = OwnershipMemory(switch_margin=0.15)
        mem.update(1, 7, 0.90)
        mem.update(1, 7, 0.30)
        assert mem.get_score(1) == pytest.approx(0.30)
        assert mem.update(1, 9, 0.50) == 9

    def test_unknown_sack_has_no_owner(self):
        mem = OwnershipMemory(switch_margin=0.15)
        assert mem.get(42) is None
        assert mem.get_score(42) == 0.0

    def test_iter_by_owner_snapshot_survives_mutation(self):
        mem = OwnershipMemory(switch_margin=0.15)
        mem.update(1, 7, 0.5)
        mem.update(2, 8, 0.5)
        for sid, _pid in mem.iter_by_owner():
            mem.clear(sid)          # mutating during iteration must be safe
        assert mem.get(1) is None and mem.get(2) is None

    def test_cleanup_keeps_only_active(self):
        mem = OwnershipMemory(switch_margin=0.15)
        mem.update(1, 7, 0.5)
        mem.update(2, 8, 0.5)
        mem.cleanup({1})
        assert mem.get(1) == 7
        assert mem.get(2) is None


# ── IdentityRelinker ──────────────────────────────────────────

class TestIdentityRelinker:
    def test_unknown_track_keeps_its_own_id(self):
        relinker = _relinker()
        got = relinker.try_relink(
            5, cx=10, cy=10, vx=0, vy=0, bbox_area=100,
            embedding=_emb(1, 0), frame_no=1, logger=_NullLogger(),
        )
        assert got == 5
        assert relinker.reid_events == []

    def test_relinks_a_close_lookalike(self):
        relinker = _relinker()
        relinker.register_lost(3, cx=100, cy=100, vx=0, vy=0,
                               bbox_area=1000, embedding=_emb(1, 0),
                               frame_no=10)
        got = relinker.try_relink(
            77, cx=105, cy=100, vx=0, vy=0, bbox_area=1000,
            embedding=_emb(1, 0), frame_no=12, logger=_NullLogger(),
        )
        assert got == 3
        assert relinker.canonical(77) == 3
        assert relinker.reid_events[0]["old_id"] == 3
        assert relinker.reid_events[0]["new_id"] == 77

    def test_does_not_relink_a_stranger(self):
        """Same place, orthogonal appearance — must stay a new person."""
        relinker = _relinker()
        relinker.register_lost(3, cx=100, cy=100, vx=0, vy=0,
                               bbox_area=1000, embedding=_emb(1, 0),
                               frame_no=10)
        got = relinker.try_relink(
            77, cx=100, cy=100, vx=0, vy=0, bbox_area=1000,
            embedding=_emb(0, 1), frame_no=12, logger=_NullLogger(),
        )
        assert got == 77
        assert relinker.reid_events == []

    def test_does_not_relink_across_the_frame(self):
        relinker = _relinker()
        relinker.register_lost(3, cx=100, cy=100, vx=0, vy=0,
                               bbox_area=1000, embedding=_emb(1, 0),
                               frame_no=10)
        got = relinker.try_relink(
            77, cx=100000, cy=100000, vx=0, vy=0, bbox_area=1000,
            embedding=_emb(1, 0), frame_no=12, logger=_NullLogger(),
        )
        assert got == 77

    def test_memory_expires(self):
        relinker = _relinker()
        relinker.set_memory_frames(5)
        relinker.register_lost(3, cx=100, cy=100, vx=0, vy=0,
                               bbox_area=1000, embedding=_emb(1, 0),
                               frame_no=10)
        got = relinker.try_relink(
            77, cx=100, cy=100, vx=0, vy=0, bbox_area=1000,
            embedding=_emb(1, 0), frame_no=100, logger=_NullLogger(),
        )
        assert got == 77

    def test_expire_lost_prunes_the_registry(self):
        relinker = _relinker()
        relinker.set_memory_frames(5)
        relinker.register_lost(3, 100, 100, 0, 0, 1000, _emb(1, 0), 10)
        relinker.expire_lost(frame_no=100)
        assert relinker.lost_registry == {}

    def test_a_lost_track_is_consumed_by_one_relink(self):
        """Two new tracks must not both inherit the same old identity."""
        relinker = _relinker()
        relinker.register_lost(3, 100, 100, 0, 0, 1000, _emb(1, 0), 10)
        first = relinker.try_relink(77, 100, 100, 0, 0, 1000,
                                    _emb(1, 0), 12, _NullLogger())
        second = relinker.try_relink(78, 100, 100, 0, 0, 1000,
                                     _emb(1, 0), 12, _NullLogger())
        assert first == 3
        assert second == 78

    def test_canonical_follows_a_chain(self):
        relinker = _relinker()
        relinker.id_map = {9: 5, 5: 2}
        assert relinker.canonical(9) == 2

    def test_canonical_survives_a_cycle(self):
        """A cycle must terminate rather than hang the frame loop."""
        relinker = _relinker()
        relinker.id_map = {1: 2, 2: 1}
        assert relinker.canonical(1) in (1, 2)

    def test_already_mapped_track_short_circuits(self):
        relinker = _relinker()
        relinker.id_map = {77: 3}
        got = relinker.try_relink(77, 0, 0, 0, 0, 0, _emb(1, 0), 5,
                                  _NullLogger())
        assert got == 3
        assert relinker.reid_events == []

    def test_register_lost_keys_on_the_canonical_id(self):
        """
        A person re-linked once and lost again must be filed under their
        original id, or the next re-link creates a second identity for
        the same worker and splits their count.
        """
        relinker = _relinker()
        relinker.id_map = {77: 3}
        relinker.register_lost(77, 100, 100, 0, 0, 1000, _emb(1, 0), 20)
        assert 3 in relinker.lost_registry
        assert 77 not in relinker.lost_registry


# ── GhostSacks ────────────────────────────────────────────────

class TestGhostSacks:
    def test_ghost_appears_when_a_sack_vanishes(self):
        ghosts = GhostSacks(max_frames=5)
        ghosts.record_position(1, 100, 100)
        ghosts.update(active_sids=set(), sack_owner={1: 7},
                      sack_positions={1: (100, 100)},
                      sack_boxes={1: (90, 90, 110, 110)})
        assert dict(ghosts.iter_ghosts())[1]["owner"] == 7

    def test_ghost_drifts_along_last_velocity(self):
        ghosts = GhostSacks(max_frames=5)
        ghosts.record_position(1, 100, 100)
        ghosts.record_position(1, 110, 100)      # vx = +10
        ghosts.update(set(), {1: 7}, {1: (110, 100)}, {1: (100, 90, 120, 110)})
        ghosts.update(set(), {1: 7}, {1: (110, 100)}, {1: (100, 90, 120, 110)})
        assert dict(ghosts.iter_ghosts())[1]["cx"] > 110

    def test_ghost_expires_and_does_not_return(self):
        """
        The occlusion budget is spent once.  This used to regenerate a
        fresh ghost every frame after the old one aged out, forever.
        """
        ghosts = GhostSacks(max_frames=2)
        ghosts.record_position(1, 100, 100)
        positions = {1: (100, 100)}
        boxes = {1: (90, 90, 110, 110)}
        for _ in range(10):
            ghosts.update(set(), {1: 7}, positions, boxes)
        assert dict(ghosts.iter_ghosts()) == {}

    def test_redetection_restores_the_budget(self):
        ghosts = GhostSacks(max_frames=2)
        ghosts.record_position(1, 100, 100)
        positions = {1: (100, 100)}
        boxes = {1: (90, 90, 110, 110)}
        for _ in range(6):
            ghosts.update(set(), {1: 7}, positions, boxes)
        ghosts.update({1}, {1: 7}, positions, boxes)      # seen again
        ghosts.update(set(), {1: 7}, positions, boxes)
        assert 1 in dict(ghosts.iter_ghosts())

    def test_visible_sack_has_no_ghost(self):
        ghosts = GhostSacks(max_frames=5)
        ghosts.record_position(1, 100, 100)
        ghosts.update({1}, {1: 7}, {1: (100, 100)}, {1: (90, 90, 110, 110)})
        assert dict(ghosts.iter_ghosts()) == {}

    def test_cleanup_drops_history_for_forgotten_ids(self):
        ghosts = GhostSacks(max_frames=5)
        ghosts.record_position(1, 100, 100)
        ghosts.record_position(2, 200, 200)
        ghosts.cleanup(known_sids={1})
        assert 1 in ghosts._prev_pos
        assert 2 not in ghosts._prev_pos
