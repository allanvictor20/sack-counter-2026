"""
test_end_to_end.py — A whole session, start to written log.

Everything else tests a stage in isolation.  Nothing exercised the loop
that joins them: class resolution, the per-frame pipeline, the frame
sink, ``should_stop``, teardown, the report, and the JSON log it writes.
That is why bugs in those seams — a constant log filename, a callback
whose second argument differed by mode — survived so long.

The model and the video are both synthetic.  ``ultralytics`` is replaced
by a stub that returns scripted detections, so this runs in about a
second and needs no weights, no GPU, and no footage.
"""

import json
import sys
import types

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from sack_counter.session_view import SessionView  # noqa: E402


# ── Stub detector ─────────────────────────────────────────────

class _Box:
    """One detection, shaped like an ultralytics box."""

    def __init__(self, tid, cls, xyxy, conf):
        self.id   = [tid]
        self.cls  = [cls]
        self.conf = [conf]
        self.xyxy = [xyxy]


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class _StubYOLO:
    """
    A person carrying one sack, walking left-to-right through a doorway.

    Class indices are named, so resolve_class_indices matches by name
    and the console's fallback warning stays quiet.
    """

    names = {0: "person", 1: "sack", 2: "box"}

    def __init__(self, *args, **kwargs):
        self._frame = 0

    def track(self, frame, **kwargs):
        self._frame += 1
        x = 80 + self._frame * 40           # marches toward the door at x=500
        person = _Box(1, 0, [x, 200, x + 100, 500], 0.95)
        sack   = _Box(2, 1, [x + 20, 250, x + 80, 310], 0.90)
        return [_Result([person, sack])]

    predict = track


@pytest.fixture
def stub_yolo(monkeypatch):
    module = types.ModuleType("ultralytics")
    module.YOLO = _StubYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return module


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A temp working directory holding a video and a calibration."""
    monkeypatch.chdir(tmp_path)

    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video),
                             cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (960, 540))
    assert writer.isOpened(), "no mp4 writer available in this build"
    for _ in range(30):
        frame = np.zeros((540, 960, 3), np.uint8)
        frame[:] = (40, 40, 40)
        writer.write(frame)
    writer.release()

    (tmp_path / "config.yaml").write_text(
        "model_path: stub.pt\n"
        "confirm_secs: 0.05\n"
        "door_polygon_points: [[500, 100], [560, 100], [560, 500], [500, 500]]\n"
        "door_room_point: [900, 300]\n",
        encoding="utf-8")
    return tmp_path


def _run(**overrides):
    from sack_counter.main import run
    kwargs = dict(source="clip.mp4", headless=True, config_path="config.yaml")
    kwargs.update(overrides)
    return run(**kwargs)


# ── The loop ──────────────────────────────────────────────────

class TestEntrySessionRuns:
    def test_completes_and_returns_a_summary(self, stub_yolo, workspace):
        summary = _run()
        assert summary["frames"] == 30
        assert summary["mode"] == "enter"
        assert summary["source"] == "clip.mp4"

    def test_writes_a_log_that_parses(self, stub_yolo, workspace):
        summary = _run()
        written = workspace / summary["log_path"]
        assert written.exists()
        data = json.loads(written.read_text(encoding="utf-8"))
        assert data["source"] == "clip.mp4"
        assert "total_sacks" in data

    def test_two_runs_leave_two_logs(self, stub_yolo, workspace):
        """
        The regression that motivated timestamped names: the second run
        used to overwrite the first, and History showed one session.
        """
        first  = _run()
        second = _run()
        assert first["log_path"] != second["log_path"]
        assert len(list(workspace.glob("delivery_log_*.json"))) == 2

    def test_history_lists_both_runs(self, stub_yolo, workspace):
        from sack_counter.web import history as history_mod
        _run()
        _run()
        assert len(history_mod.list_sessions(workspace)) == 2


# ── Host hooks ────────────────────────────────────────────────

class TestHostHooks:
    def test_frame_sink_receives_a_session_view(self, stub_yolo, workspace):
        seen = []
        _run(frame_sink=seen.append)

        assert len(seen) == 30
        view = seen[-1]
        assert isinstance(view, SessionView)
        assert view.mode == "enter"
        assert view.frame_no == 30
        assert view.frame.shape == (540, 960, 3)

    def test_view_exposes_totals_and_workers(self, stub_yolo, workspace):
        seen = []
        _run(frame_sink=seen.append)
        view = seen[-1]
        assert view.totals["label"] == "Sacks in"
        assert isinstance(view.workers, list)
        assert isinstance(view.events, list)

    def test_should_stop_ends_the_run_cleanly(self, stub_yolo, workspace):
        """
        Stopping must still produce a report and a log — the web console's
        "End session" button depends on it.
        """
        calls = {"n": 0}

        def stop():
            calls["n"] += 1
            return calls["n"] > 10

        summary = _run(should_stop=stop)
        assert summary["frames"] < 30
        assert (workspace / summary["log_path"]).exists()

    def test_stopping_before_the_first_frame_still_reports(self, stub_yolo,
                                                           workspace):
        summary = _run(should_stop=lambda: True)
        assert summary["frames"] == 0
        assert (workspace / summary["log_path"]).exists()


# ── Configuration is honoured ─────────────────────────────────

class TestConfiguration:
    def test_confidence_override_reaches_the_config(self, stub_yolo,
                                                    workspace):
        seen = []
        _run(conf_sack=0.9, frame_sink=seen.append)
        assert seen[-1].session.cfg["conf_sack"] == 0.9

    def test_unnamed_classes_raise_the_console_warning(self, stub_yolo,
                                                       workspace,
                                                       monkeypatch):
        monkeypatch.setattr(_StubYOLO, "names", {0: "a", 1: "b", 2: "c"})
        seen = []
        _run(frame_sink=seen.append)
        assert "could not be matched" in (seen[-1].warning or "")

    def test_named_classes_raise_no_warning(self, stub_yolo, workspace):
        seen = []
        _run(frame_sink=seen.append)
        assert seen[-1].warning is None

    def test_save_output_writes_a_video(self, stub_yolo, workspace):
        _run(save_output=True)
        assert list(workspace.glob("output_*.mp4"))
