"""
test_config.py — Config loading, the calibration sidecar, log naming.

The calibration round-trip is the part that matters: an interactive run
clicks a doorway, saves it, and ``--headless`` has to replay exactly that
polygon on the next invocation.  That path crosses YAML serialisation of
numpy scalars, sidecar merging precedence, and file encoding, and none
of it was covered.

Every test runs in its own ``tmp_path`` because both ``load_config`` and
``save_calibration`` resolve their files relative to the process's
working directory.
"""

import json

import pytest

from sack_counter.config import (
    CALIBRATION_FILE, DEFAULT_CFG, load_config, save_calibration,
)
from sack_counter.session_log import session_log_path
from sack_counter.version import VERSION_TAG


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ── Loading ───────────────────────────────────────────────────

class TestLoadConfig:
    def test_defaults_when_nothing_present(self):
        cfg = load_config()
        assert cfg["conf_sack"] == DEFAULT_CFG["conf_sack"]
        assert cfg["model_path"] == DEFAULT_CFG["model_path"]

    def test_missing_keys_fall_back_to_defaults(self, tmp_path):
        (tmp_path / "partial.yaml").write_text("conf_sack: 0.9\n",
                                               encoding="utf-8")
        cfg = load_config("partial.yaml")
        assert cfg["conf_sack"] == 0.9
        assert cfg["conf_person"] == DEFAULT_CFG["conf_person"]

    def test_json_config_is_accepted(self, tmp_path):
        (tmp_path / "c.json").write_text(json.dumps({"conf_box": 0.11}),
                                         encoding="utf-8")
        assert load_config("c.json")["conf_box"] == 0.11

    def test_picks_up_config_yaml_from_cwd(self, tmp_path):
        (tmp_path / "config.yaml").write_text("conf_sack: 0.42\n",
                                              encoding="utf-8")
        assert load_config()["conf_sack"] == 0.42

    def test_missing_file_falls_back_without_raising(self):
        cfg = load_config("nope.yaml")
        assert cfg["conf_sack"] == DEFAULT_CFG["conf_sack"]

    def test_non_ascii_config_survives(self, tmp_path):
        """
        open() defaults to the locale codepage on Windows, so a config
        with a non-cp1252 byte in a comment used to raise mid-parse and
        silently revert every setting to defaults.
        """
        (tmp_path / "c.yaml").write_text(
            "# threshold — tuned for the café door\nconf_sack: 0.77\n",
            encoding="utf-8")
        assert load_config("c.yaml")["conf_sack"] == 0.77

    def test_malformed_yaml_falls_back_loudly(self, tmp_path, capsys):
        (tmp_path / "bad.yaml").write_text("conf_sack: [unclosed\n",
                                           encoding="utf-8")
        cfg = load_config("bad.yaml")
        assert cfg["conf_sack"] == DEFAULT_CFG["conf_sack"]
        assert "reverted to built-in defaults" in capsys.readouterr().out


# ── Calibration sidecar ───────────────────────────────────────

class TestCalibrationRoundTrip:
    def test_save_then_load_replays_the_polygon(self):
        cfg = load_config()
        cfg["door_polygon_points"] = [(10, 10), (90, 10), (90, 90), (10, 90)]
        cfg["door_room_point"] = (50, 150)
        save_calibration(cfg)

        replayed = load_config()
        assert replayed["door_polygon_points"] == [[10, 10], [90, 10],
                                                   [90, 90], [10, 90]]
        assert replayed["door_room_point"] == [50, 150]

    def test_numpy_scalars_serialise_as_plain_ints(self, tmp_path):
        """
        DoorPolygon.points come out of cv2.convexHull as numpy int32.
        Dumped raw they become !!python/object tags that safe_load then
        refuses to read back — a calibration that saves but never loads.
        """
        np = pytest.importorskip("numpy")
        cfg = load_config()
        cfg["door_polygon_points"] = [(np.int32(1), np.int32(2))]
        save_calibration(cfg)

        text = (tmp_path / CALIBRATION_FILE).read_text(encoding="utf-8")
        assert "!!python" not in text
        assert load_config()["door_polygon_points"] == [[1, 2]]

    def test_explicit_config_wins_over_the_sidecar(self, tmp_path):
        """
        A hand-pinned polygon must not be silently replaced by a stale
        saved calibration.
        """
        cfg = load_config()
        cfg["door_polygon_points"] = [(1, 1), (2, 1), (2, 2), (1, 2)]
        save_calibration(cfg)

        (tmp_path / "pinned.yaml").write_text(
            "door_polygon_points: [[9, 9], [8, 9], [8, 8], [9, 8]]\n",
            encoding="utf-8")
        assert load_config("pinned.yaml")["door_polygon_points"] == [
            [9, 9], [8, 9], [8, 8], [9, 8]]

    def test_sidecar_fills_only_calibration_keys(self, tmp_path):
        cfg = load_config()
        cfg["door_polygon_points"] = [(1, 1), (2, 1), (2, 2), (1, 2)]
        cfg["conf_sack"] = 0.99            # not a calibration key
        save_calibration(cfg)

        text = (tmp_path / CALIBRATION_FILE).read_text(encoding="utf-8")
        assert "conf_sack" not in text
        assert load_config()["conf_sack"] == DEFAULT_CFG["conf_sack"]

    def test_nothing_to_save_writes_no_file(self, tmp_path):
        assert save_calibration(load_config()) is None
        assert not (tmp_path / CALIBRATION_FILE).exists()

    def test_unreadable_sidecar_is_ignored_not_fatal(self, tmp_path):
        (tmp_path / CALIBRATION_FILE).write_text("{[bad\n", encoding="utf-8")
        assert load_config()["conf_sack"] == DEFAULT_CFG["conf_sack"]


# ── Session log naming ────────────────────────────────────────

class TestSessionLogPath:
    def test_carries_prefix_and_version(self):
        name = session_log_path("delivery_log")
        assert name.startswith(f"delivery_log_{VERSION_TAG}_")
        assert name.endswith(".json")

    def test_two_sessions_do_not_collide(self):
        """
        Entry mode used a constant name, so each run overwrote the last
        and History could only ever list one entry session.
        """
        from datetime import datetime
        first  = session_log_path("delivery_log", datetime(2026, 8, 4, 9, 0, 0))
        second = session_log_path("delivery_log", datetime(2026, 8, 4, 9, 0, 1))
        assert first != second

    def test_same_second_still_does_not_collide(self, tmp_path):
        """
        The timestamp has one-second resolution, so two sessions ending
        inside the same second would otherwise overwrite each other —
        the same bug, just rarer and harder to notice.
        """
        from datetime import datetime
        when = datetime(2026, 8, 4, 9, 0, 0)
        first = session_log_path("delivery_log", when, directory=tmp_path)
        (tmp_path / first).write_text("{}", encoding="utf-8")
        second = session_log_path("delivery_log", when, directory=tmp_path)
        assert second != first
        assert second.endswith(".json")

    def test_still_matches_the_history_glob(self):
        import fnmatch
        assert fnmatch.fnmatch(session_log_path("delivery_log"),
                               "delivery_log_*.json")
        assert fnmatch.fnmatch(session_log_path("exit_log"),
                               "exit_log_*.json")

    def test_names_sort_chronologically(self):
        from datetime import datetime
        early = session_log_path("exit_log", datetime(2026, 1, 2, 3, 4, 5))
        late  = session_log_path("exit_log", datetime(2026, 11, 12, 13, 14, 15))
        assert early < late
