"""
test_web.py — The console's routes, guards and log parsing.

The web layer had no tests at all, including every validation guard —
the ones that stop a session starting against a missing file or an
uncalibrated camera, and the path handling on the report route.

Nothing here starts a real session: that needs a model and a video, and
is covered by the end-to-end test instead.  These cover the HTTP surface
around it.
"""

import json

import pytest

fastapi_testclient = pytest.importorskip(
    "fastapi.testclient", reason="web console extra not installed")
TestClient = fastapi_testclient.TestClient

from sack_counter.web import history as history_mod  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A client whose working directory is an empty temp dir."""
    monkeypatch.chdir(tmp_path)
    from sack_counter.web.app import app
    return TestClient(app)


@pytest.fixture
def calibrated(tmp_path):
    """A saved calibration, so start-session guards get past the door check."""
    (tmp_path / "calibration.yaml").write_text(
        "door_polygon_points:\n"
        "- [100, 100]\n- [300, 100]\n- [300, 300]\n- [100, 300]\n"
        "door_room_point: [200, 400]\n",
        encoding="utf-8")
    return tmp_path


# ── Screens ───────────────────────────────────────────────────

class TestScreensRender:
    @pytest.mark.parametrize("path", [
        "/", "/setup", "/calibrate", "/history", "/diagnostics",
    ])
    def test_screen_returns_html(self, client, path):
        response = client.get(path)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    @pytest.mark.parametrize("path", [
        "/", "/setup", "/calibrate", "/history", "/diagnostics",
    ])
    def test_no_unrendered_template_tags(self, client, path):
        """A stray {{ }} means a context key the view forgot to pass."""
        body = client.get(path).text
        assert "{{" not in body
        assert "{%" not in body

    def test_report_with_no_logs_still_renders(self, client):
        assert client.get("/report").status_code == 200

    def test_unknown_report_id_is_404(self, client):
        assert client.get("/report/nope.json").status_code == 404

    def test_report_id_cannot_escape_the_working_directory(self, client):
        """
        A log id is a file name, never a path the caller can steer.
        Traversal must not read outside the working directory.
        """
        response = client.get("/report/..%2F..%2Fconfig.yaml")
        assert response.status_code in (404, 400)


# ── Session guards ────────────────────────────────────────────

class TestSessionStartGuards:
    def test_empty_source_is_rejected(self, client):
        response = client.post("/api/session/start", json={"source": ""})
        assert response.status_code == 400
        assert "Choose a video" in response.json()["detail"]

    def test_missing_file_is_rejected(self, client):
        response = client.post("/api/session/start",
                               json={"source": "no_such_video.mp4"})
        assert response.status_code == 400
        assert "No such video file" in response.json()["detail"]

    def test_uncalibrated_camera_is_rejected(self, client, tmp_path):
        (tmp_path / "clip.mp4").write_bytes(b"not really a video")
        response = client.post("/api/session/start",
                               json={"source": "clip.mp4"})
        assert response.status_code == 400
        assert "not calibrated" in response.json()["detail"]

    def test_exit_mode_needs_a_landing_zone(self, client, calibrated):
        (calibrated / "clip.mp4").write_bytes(b"not really a video")
        response = client.post("/api/session/start",
                               json={"source": "clip.mp4", "mode": "exit"})
        assert response.status_code == 400
        assert "landing zone" in response.json()["detail"]

    def test_snapshot_is_idle_before_any_session(self, client):
        body = client.get("/api/session").json()
        assert body["running"] is False
        assert body["status"] == "idle"

    def test_stop_with_nothing_running_is_harmless(self, client):
        assert client.post("/api/session/stop").status_code == 200


# ── Calibration API ───────────────────────────────────────────

class TestCalibrationApi:
    def test_saves_a_valid_doorway(self, client, tmp_path):
        response = client.post("/api/calibration", json={
            "points": [[100, 100], [300, 100], [300, 300], [100, 300]],
            "room_point": [200, 400],
        })
        assert response.status_code == 200
        assert (tmp_path / "calibration.yaml").exists()

    def test_rejects_the_wrong_number_of_corners(self, client):
        response = client.post("/api/calibration", json={
            "points": [[0, 0], [1, 1]], "room_point": [5, 5]})
        assert response.status_code == 400

    def test_rejects_a_missing_room_point(self, client):
        response = client.post("/api/calibration", json={
            "points": [[100, 100], [300, 100], [300, 300], [100, 300]]})
        assert response.status_code == 400

    def test_rejects_a_degenerate_polygon(self, client):
        """Collinear corners must be refused here, not at the next run."""
        response = client.post("/api/calibration", json={
            "points": [[0, 0], [10, 0], [20, 0], [30, 0]],
            "room_point": [15, 50]})
        assert response.status_code == 400

    def test_landing_zone_needs_three_points(self, client):
        response = client.post("/api/calibration", json={
            "target": "landing", "landing_points": [[0, 0], [10, 10]]})
        assert response.status_code == 400

    def test_landing_zone_caps_at_eight_points(self, client):
        response = client.post("/api/calibration", json={
            "target": "landing",
            "landing_points": [[i, i] for i in range(9)]})
        assert response.status_code == 400

    def test_settings_round_trip(self, client):
        response = client.post("/api/settings", json={"conf_sack": 0.5})
        assert response.status_code == 200
        assert response.json()["cfg"]["conf_sack"] == 0.5


# ── Diagnostics guards ────────────────────────────────────────

class TestDiagnosticsApi:
    def test_source_is_required(self, client):
        response = client.post("/api/diagnostics/start", json={"source": ""})
        assert response.status_code == 400

    def test_cutoff_must_be_a_number(self, client):
        response = client.post("/api/diagnostics/start",
                               json={"source": "clip.mp4", "cutoff": "abc"})
        assert response.status_code == 400

    def test_cutoff_must_be_in_range(self, client):
        response = client.post("/api/diagnostics/start",
                               json={"source": "clip.mp4", "cutoff": 1.5})
        assert response.status_code == 400

    def test_snapshot_available_before_any_run(self, client):
        assert client.get("/api/diagnostics").status_code == 200


# ── History parsing ───────────────────────────────────────────

class TestHistory:
    def _write(self, directory, name, payload):
        (directory / name).write_text(json.dumps(payload), encoding="utf-8")

    def test_lists_both_modes(self, tmp_path):
        self._write(tmp_path, "delivery_log_v23_20260101_000000.json",
                    {"mode": "enter", "total_sacks": 5, "source": "a.mp4"})
        self._write(tmp_path, "exit_log_v23_20260101_000001.json",
                    {"mode": "exit", "sacks_exited": 9, "source": "b.mp4"})
        rows = history_mod.list_sessions(tmp_path)
        assert {row["mode"] for row in rows} == {"enter", "exit"}

    def test_two_entry_sessions_both_appear(self, tmp_path):
        """
        The whole point of timestamping the log name: a second entry run
        must not replace the first one's record.
        """
        self._write(tmp_path, "delivery_log_v23_20260101_000000.json",
                    {"mode": "enter", "total_sacks": 5})
        self._write(tmp_path, "delivery_log_v23_20260101_010000.json",
                    {"mode": "enter", "total_sacks": 8})
        assert len(history_mod.list_sessions(tmp_path)) == 2

    def test_mode_inferred_from_the_filename(self, tmp_path):
        self._write(tmp_path, "exit_log_v23_20260101_000000.json",
                    {"sacks_exited": 3})
        assert history_mod.list_sessions(tmp_path)[0]["mode"] == "exit"

    def test_corrupt_log_is_skipped_not_fatal(self, tmp_path):
        (tmp_path / "delivery_log_v23_bad.json").write_text("{[",
                                                            encoding="utf-8")
        self._write(tmp_path, "delivery_log_v23_ok.json",
                    {"mode": "enter", "total_sacks": 1})
        assert len(history_mod.list_sessions(tmp_path)) == 1

    def test_sparse_log_still_yields_every_field(self, tmp_path):
        self._write(tmp_path, "delivery_log_v23_20260101_000000.json", {})
        row = history_mod.list_sessions(tmp_path)[0]
        for key in ("id", "mode", "sacks", "boxes", "workers", "when_iso"):
            assert key in row

    def test_empty_directory_lists_nothing(self, tmp_path):
        assert history_mod.list_sessions(tmp_path) == []

    def test_load_report_of_a_missing_file_is_none(self, tmp_path):
        assert history_mod.load_report(tmp_path / "nope.json") is None

    def test_load_report_builds_a_summary(self, tmp_path):
        self._write(tmp_path, "delivery_log_v23_20260101_000000.json",
                    {"mode": "enter", "total_sacks": 7, "total_boxes": 2})
        report = history_mod.load_report(
            tmp_path / "delivery_log_v23_20260101_000000.json")
        assert report is not None
        assert report["summary"]["display_label"] == "Sacks Entered"
