"""
config.py — Configuration loader for Sack Counter v20.

Supports both YAML (preferred) and JSON config files.
Falls back to DEFAULT_CFG for any missing keys.
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
    "conf_sack":             0.35,
    "conf_person":           0.55,
    "conf_box":              0.45,
    "confirm_secs":          0.4,
    "miss_secs":             1.5,
    "occlusion_secs":        0.5,
    "still_window":          10,
    "still_thresh_ratio":    0.02,
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
    "peak_min_assoc_score":     0.20,
    "peak_conf_sack_floor":     0.50,
    "still_suppress_secs":      1.5,
    "lift_owner_stable_frames": 6,
}


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
        return cfg

    p = Path(path)
    if not p.exists():
        print(f"  Warning: config file '{path}' not found — using defaults.")
        return cfg

    try:
        if p.suffix in (".yaml", ".yml"):
            if not _YAML_AVAILABLE:
                raise ImportError(
                    "PyYAML is not installed. Run: pip install pyyaml"
                )
            with open(p) as f:
                overrides = yaml.safe_load(f) or {}
        else:
            with open(p) as f:
                overrides = json.load(f)

        cfg.update(overrides)
        print(f"  Config loaded from {path}")
    except Exception as exc:
        print(f"  Warning: could not load '{path}': {exc} — using defaults.")

    return cfg
