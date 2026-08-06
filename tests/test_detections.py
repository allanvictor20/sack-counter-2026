"""
test_detections.py — Turning tracker output into pipeline tuples.

This was inline in the frame loop, so the per-class confidence floors
could only be checked by running a model over a video.  It is a pure
function now, and these are the cases that decide whether a detection
reaches the pipeline at all.
"""

import pytest

from sack_counter.config import DEFAULT_CFG
from sack_counter.detections import detection_conf_floor, parse_detections

CLS = {"person": 0, "sack": 1, "box": 2}


class _Det:
    def __init__(self, tid, cls, xyxy, conf):
        self.id   = None if tid is None else [tid]
        self.cls  = [cls]
        self.conf = [conf]
        self.xyxy = [xyxy]


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


@pytest.fixture
def cfg():
    return dict(DEFAULT_CFG, conf_person=0.55, conf_sack=0.35, conf_box=0.45)


class TestParseDetections:
    def test_splits_by_class(self, cfg):
        got = parse_detections([_Result([
            _Det(1, 0, [0, 0, 10, 20], 0.9),
            _Det(2, 1, [5, 5, 15, 15], 0.9),
            _Det(3, 2, [7, 7, 17, 17], 0.9),
        ])], CLS, cfg)
        assert len(got.persons) == 1
        assert len(got.sacks) == 1
        assert len(got.boxes) == 1

    def test_sack_tuple_carries_its_centroid(self, cfg):
        """Assignment and crossing both need it, so it is computed once here."""
        got = parse_detections([_Result([_Det(2, 1, [0, 0, 100, 50], 0.9)])],
                               CLS, cfg)
        sid, x1, y1, x2, y2, cx, cy, score = got.sacks[0]
        assert (cx, cy) == (50, 25)

    def test_untracked_detection_is_dropped(self, cfg):
        """Everything downstream is keyed on a stable id."""
        got = parse_detections([_Result([_Det(None, 1, [0, 0, 10, 10], 0.99)])],
                               CLS, cfg)
        assert got.sacks == []

    @pytest.mark.parametrize("cls,score,field", [
        (0, 0.54, "persons"),
        (1, 0.34, "sacks"),
        (2, 0.44, "boxes"),
    ])
    def test_below_its_own_floor_is_dropped(self, cfg, cls, score, field):
        got = parse_detections([_Result([_Det(1, cls, [0, 0, 10, 10], score)])],
                               CLS, cfg)
        assert getattr(got, field) == []

    @pytest.mark.parametrize("cls,score,field", [
        (0, 0.55, "persons"),
        (1, 0.35, "sacks"),
        (2, 0.45, "boxes"),
    ])
    def test_exactly_at_its_floor_is_kept(self, cfg, cls, score, field):
        got = parse_detections([_Result([_Det(1, cls, [0, 0, 10, 10], score)])],
                               CLS, cfg)
        assert len(getattr(got, field)) == 1

    def test_floors_are_per_class_not_shared(self, cfg):
        """
        A 0.40 sack must survive while a 0.40 person does not — the whole
        reason the detector runs at the minimum and filters afterwards.
        """
        got = parse_detections([_Result([
            _Det(1, 0, [0, 0, 10, 10], 0.40),
            _Det(2, 1, [0, 0, 10, 10], 0.40),
        ])], CLS, cfg)
        assert got.persons == []
        assert len(got.sacks) == 1

    def test_respects_a_reordered_model(self, cfg):
        """Class indices come from name resolution, never position."""
        reordered = {"person": 2, "sack": 0, "box": 1}
        got = parse_detections([_Result([_Det(1, 2, [0, 0, 10, 10], 0.9)])],
                               reordered, cfg)
        assert len(got.persons) == 1
        assert got.sacks == []

    def test_unknown_class_index_is_ignored(self, cfg):
        got = parse_detections([_Result([_Det(1, 7, [0, 0, 10, 10], 0.99)])],
                               CLS, cfg)
        assert (got.persons, got.sacks, got.boxes) == ([], [], [])

    def test_empty_and_missing_results_are_safe(self, cfg):
        assert parse_detections([], CLS, cfg).sacks == []
        assert parse_detections(None, CLS, cfg).sacks == []
        assert parse_detections([_Result(None)], CLS, cfg).sacks == []

    def test_coordinates_become_ints(self, cfg):
        got = parse_detections([_Result([_Det(1, 1, [1.7, 2.2, 9.9, 8.1],
                                              0.9)])], CLS, cfg)
        assert all(isinstance(v, int) for v in got.sacks[0][1:5])


class TestDetectionConfFloor:
    def test_is_the_lowest_of_the_three(self, cfg):
        assert detection_conf_floor(cfg) == 0.35

    def test_tracks_the_lowest_when_it_changes(self, cfg):
        cfg["conf_box"] = 0.10
        assert detection_conf_floor(cfg) == 0.10
