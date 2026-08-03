"""
config.py — Configuration loader for Sack Counter v23.

Supports both YAML (preferred) and JSON config files.
Falls back to DEFAULT_CFG for any missing keys.

DEFAULT_CFG is the single source of truth for every tunable in the
system, including exit-mode keys.  Modules must not invent their own
inline fallbacks via ``cfg.get(key, some_other_default)`` — that is how
``door_cross_threshold_px`` ended up meaning 30 in entry mode and 20 in
exit mode.  Use ``cfg[key]`` and add the key here instead.
"""

import json
from pathlib import Path
from typing import Any

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


DEFAULT_CFG: dict[str, Any] = {
    "model_path":            "best.pt",
    # Exit mode may use a separate model trained on ground-lying sacks.
    # Null / missing falls back to model_path.
    "exit_model_path":       None,
    "conf_sack":             0.35,
    "conf_person":           0.55,
    "conf_box":              0.45,
    "confirm_secs":          0.4,
    "miss_secs":             1.5,
    "occlusion_secs":        0.5,
    "w_distance":            0.35,
    "w_overlap":             0.25,
    "w_height":              0.20,
    "w_appearance":          0.20,
    "carry_zone_top":       -1.5,
    "carry_zone_bot":        0.6,
    "max_assoc_dist":        300,
    "id_jump_ratio":         0.25,
    "min_assoc_score":       0.05,
    "max_sacks_per_person":  8,
    "reid_memory_secs":      3.0,
    "reid_threshold":        0.72,
    "reid_w_position":       0.25,
    "reid_w_velocity":       0.15,
    "reid_w_bbox_size":      0.10,
    "reid_w_appearance":     0.50,
    "still_speed_thresh":    1.5,
    "still_speed_window":    8,
    "ownership_switch_margin": 0.15,
    # Door polygon (v22 — replaces gate)
    "door_polygon_points":        None,     # set by calibration
    "door_room_point":            None,     # room-side click (v22)
    "door_disappear_frames":      15,
    "commit_delay_frames":        10,       # v22: extra frames before commit
    "door_entry_proximity_px":    60,       # FIX-D: px from polygon plane to trigger DOOR-ENTER
    "max_candidate_age_frames":   300,      # v22: evict stale candidates
    "direction_lookback_frames":  8,
    "door_normal_threshold":      0.0,      # v22: min dot product for direction
    "min_convergence_px":         20,       # kept for reference (unused in v22)
    "reentry_margin_px":          50,
    "reentry_grace_frames":       5,        # frames after a commit during
                                            # which re-entry cannot reset
    "door_cross_threshold_px":    5,        # signed projection past the door
                                            # plane that counts as "crossed"
    "door_approach_side":         "right",
    "polygon_padding_px":         0,        # optional padding around polygon
    "minimum_visible_frames":     3,        # min frames inside polygon before tracking
    "conf_weight_detection":  0.30,
    "conf_weight_tracking":   0.30,
    "conf_weight_ownership":  0.25,
    "conf_weight_crossing":   0.15,
    "min_delivery_confidence": 0.55,
    "ownership_anomaly_threshold": 0.50,
    "conf_class_high":   0.80,
    "conf_class_medium": 0.60,
    "peak_count_enabled":       True,
    "peak_window_px":           350,
    "peak_freeze_px":           40,
    "peak_min_assoc_score":     0.55,
    "peak_conf_sack_floor":     0.55,
    # Association score a sack must reach before it is stamped to a
    # carrier.  Previously hardcoded as max(peak_min_assoc_score + 0.1,
    # 0.6), which silently ignored any configured value below 0.6.
    "peak_stamp_score":         0.65,
    "still_suppress_secs":      1.5,
    "lift_owner_stable_frames": 6,

    # ── Exit mode (v23) ───────────────────────────────────────────
    "landing_zone_points":            None,   # set by exit calibration
    "exit_confirm_timeout_frames":    45,
    "exit_still_frames":              8,
    "exit_landing_miss_grace_frames": 5,
    "exit_dedup_radius_px":           35,
    "exit_dedup_recency_window":      20,
    "exit_landing_jitter_tolerance":  2,
    "exit_landing_min_overlap":       0.30,   # fraction of a sack bbox that
                                              # must fall inside the zone
    "exit_stale_tentative_frames":    90,
}


# Keys whose value is produced by interactive calibration.  ``save_config``
# writes these back so ``--headless`` can replay a calibrated session.
CALIBRATION_KEYS = (
    "door_polygon_points",
    "door_room_point",
    "door_approach_side",
    "landing_zone_points",
)

# Sidecar written by save_calibration() and merged by load_config().
# Kept separate from config.yaml so saving a calibration never rewrites
# (and thereby strips the comments from) the hand-edited config file.
CALIBRATION_FILE = "calibration.yaml"


def load_config(path: str | None = None) -> dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.

    Falls back to DEFAULT_CFG for any missing keys.
    If *path* is None, checks for ``config.yaml`` in the current
    directory before falling back to defaults.

    Args:
        path: Path to a ``.yaml`` or ``.json`` config file, or None.

    Returns:
        Merged configuration dictionary.
    """
    cfg = dict(DEFAULT_CFG)

    if path is None:
        candidate = Path("config.yaml")
        if candidate.exists():
            path = str(candidate)

    if path is None:
        _merge_calibration(cfg)
        return cfg

    p = Path(path)
    if not p.exists():
        print(f"  Warning: config file '{path}' not found — using defaults.")
        _merge_calibration(cfg)
        return cfg

    try:
        # encoding="utf-8" is required, not cosmetic.  open() defaults to
        # the locale codepage (cp1252 on a stock Windows install), so a
        # config containing any byte undefined in that codepage raises
        # UnicodeDecodeError — which the except below swallows into a
        # single warning line while EVERY setting silently reverts to
        # DEFAULT_CFG.  Losing a whole tuned config to a stray character
        # in a comment is not a failure mode worth keeping.
        if p.suffix in (".yaml", ".yml"):
            if not _YAML_AVAILABLE:
                raise ImportError(
                    "PyYAML is not installed. Run: pip install pyyaml"
                )
            with open(p, encoding="utf-8") as f:
                overrides = yaml.safe_load(f) or {}
        else:
            with open(p, encoding="utf-8") as f:
                overrides = json.load(f)

        cfg.update(overrides)
        print(f"  Config loaded from {path}")
    except Exception as exc:
        print(f"  Warning: could not load '{path}': {exc} — using defaults.")
        print("  !! All settings have reverted to built-in defaults. "
              "Fix the config file before trusting any counts. !!")

    _merge_calibration(cfg)
    return cfg


def _merge_calibration(cfg: dict[str, Any]) -> None:
    """
    Overlay the calibration sidecar (if present) onto *cfg* in place.

    Values already set explicitly in the user's config file win — the
    sidecar only fills in calibration keys that are still None, so a
    hand-pinned polygon in config.yaml is never silently overridden by a
    stale saved calibration.
    """
    sidecar = Path(CALIBRATION_FILE)
    if not sidecar.exists() or not _YAML_AVAILABLE:
        return
    try:
        with open(sidecar, encoding="utf-8") as f:
            saved = yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"  Warning: could not read '{sidecar}': {exc} — ignoring.")
        return

    applied = [k for k in CALIBRATION_KEYS
               if cfg.get(k) is None and saved.get(k) is not None]
    for key in applied:
        cfg[key] = saved[key]
    if applied:
        print(f"  Calibration loaded from {sidecar} ({', '.join(applied)})")


def _plain(value: Any) -> Any:
    """
    Convert numpy scalars / arrays and tuples into plain Python types.

    ``DoorPolygon.points`` holds numpy int32 values (they come out of
    ``cv2.convexHull``), which neither ``json`` nor ``yaml`` can serialise
    as ordinary scalars.  Without this, saving a calibration either raised
    or produced a YAML file full of ``!!python/object`` tags that
    ``yaml.safe_load`` then refused to read back.
    """
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    item = getattr(value, "item", None)          # numpy scalar
    if callable(item) and not isinstance(value, (str, bytes)):
        try:
            return item()
        except (ValueError, TypeError):
            pass
    return value


def save_calibration(cfg: dict[str, Any], path: str | None = None) -> str | None:
    """
    Write the calibration keys from *cfg* to the calibration sidecar file.

    Interactive calibration used to update only the in-memory ``cfg``
    dict, so the clicked door/landing polygons were discarded on exit and
    ``--headless`` could never be satisfied without hand-editing YAML.

    The values are written to :data:`CALIBRATION_FILE` rather than back
    into ``config.yaml``, because rewriting the latter through
    ``yaml.safe_dump`` would strip every explanatory comment in it.
    :func:`load_config` merges the sidecar automatically, so a calibrated
    run is replayable with ``--headless`` on the next invocation.

    Args:
        cfg:  Config dict containing calibrated values.
        path: Target file.  Defaults to :data:`CALIBRATION_FILE`.

    Returns:
        The path written, or None if there was nothing to save / it failed.
    """
    target = Path(path or CALIBRATION_FILE)
    updates = {k: _plain(cfg[k]) for k in CALIBRATION_KEYS if cfg.get(k) is not None}
    if not updates:
        return None

    try:
        with open(target, "w", encoding="utf-8") as f:
            if target.suffix in (".yaml", ".yml"):
                if not _YAML_AVAILABLE:
                    raise ImportError("PyYAML is not installed.")
                f.write("# Auto-generated by Sack Counter calibration.\n")
                f.write("# Delete this file to force re-calibration.\n")
                yaml.safe_dump(updates, f, sort_keys=False, default_flow_style=False)
            else:
                json.dump(updates, f, indent=2)
        print(f"  Calibration saved to {target} — --headless can now replay it.")
        return str(target)
    except Exception as exc:
        print(f"  Warning: could not save calibration to '{target}': {exc}")
        return None