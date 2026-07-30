
---

# v22 — Robustness Overhaul

## Mandatory changes (all implemented)

### 1. Normal-vector geometry (`door_polygon.py`)
- `DoorPolygon` now stores `normal_vec`, `corridor_midpt`, and `room_midpt`.
- All direction and side queries use `project_onto_normal()` instead of
  Euclidean distance to centroid.  This eliminates false positives when a
  carrier approaches diagonally or from the side.
- `movement_toward_door(old_x, old_y, cur_x, cur_y)` replaces the old
  centroid-distance convergence check in direction confirmation.
- `distance_to_centroid()` kept for backward compatibility but no longer
  used internally.

### 2. Room-side calibration click (`door_polygon.py`, `main.py`)
- Calibration now has two phases:
  - Phase 1: click 4 door corners (unchanged).
  - Phase 2: click once **inside the room** to set the room direction.
- `DoorPolygon(room_point=...)` uses this point to orient `normal_vec`
  toward the room, removing the "top edge = room side" assumption.
- Headless mode reads `door_room_point` from `config.yaml`.

### 3. Polygon validation (`door_polygon.py`, `calibrate_door_polygon()`)
- After 4 clicks, a convex hull is computed and validated with
  `cv2.isContourConvex`.
- An X-shaped (self-intersecting) polygon now raises `ValueError` with
  a user-readable message and prompts the user to redraw.
- Points are automatically sorted clockwise so click order no longer matters.

### 4. Normal-based direction confirmation (`door_crossing.py`)
- `_confirm_direction()` now calls `door.movement_toward_door()`, which
  computes `dot(movement_vector, door.normal_vec)`.  If the dot product
  exceeds `door_normal_threshold` (default 0.0), direction is confirmed.
- Old centroid-distance logic removed.

### 5. Normal-projected peak windows (`counting.py`)
- `update_peak_counts()` now computes the approach window in **portal
  coordinates** (projection onto `normal_vec`) instead of screen-space
  x coordinates.  This makes the window camera-angle independent.
- `approach_open_proj = -peak_window_px` (corridor side).
- `freeze_proj = +peak_freeze_px` (just inside the door toward room).
- Falls back to x-coordinate logic if `normal_vec` is unavailable.

### 6. History cleanup / memory leak prevention (`person_tracker.py`, `door_crossing.py`)
- `cleanup_door_histories(pid, state)` added to `door_crossing.py`.
- Called by `timeout_persons()` every time a person is evicted.
- Cleans `person_position_history`, `door_candidates`, and `person_peak_count`
  for the evicted pid, preventing unbounded growth on long videos.

## Highly beneficial changes (all implemented)

### 7. Explicit DoorState enum (`door_crossing.py`)
- `DoorState` enum: `CORRIDOR → APPROACHING → IN_DOOR → DISAPPEARING → COUNTED / ABORTED`.
- Every candidate stores `door_state: DoorState`.
- Log lines include the state name, e.g. `DOOR-DISAPPEARING P#12 state=IN_DOOR`.

### 8. Two-phase commit (`door_crossing.py`, `config.py`)
- Phase 1 (DISAPPEARING): carrier missing ≥ `door_disappear_frames`.
- Phase 2 (commit): carrier still missing `commit_delay_frames` later.
- Default: 15 + 10 = 25 frames before a delivery is counted.
- Dramatically reduces false positives from brief tracker failures.
- New config key: `commit_delay_frames` (default 10).

### 9. Stale candidate eviction (`door_crossing.py`, `config.py`)
- Candidates older than `max_candidate_age_frames` (default 300) are
  automatically removed.
- Prevents unbounded candidate accumulation when a tracker bug causes a
  carrier to never complete the sequence.
- New config key: `max_candidate_age_frames` (default 300).

### 10. Enhanced debug overlays (`door_polygon.py`)
- `draw()` now draws:
  - A green arrowed line showing the normal vector (corridor → room direction).
  - "CORRIDOR" and "ROOM" labels on each side of the door.
  - Per-candidate dot + state label at their last known position.
- Controlled by `show_debug=True` (default).

## New config keys (v22)

| Key                       | Default | Description                                         |
|---------------------------|---------|-----------------------------------------------------|
| `door_room_point`         | `None`  | (x, y) clicked inside the room during calibration  |
| `commit_delay_frames`     | `10`    | Extra frames to wait after DISAPPEARING before commit|
| `max_candidate_age_frames`| `300`   | Evict candidates older than this                    |
| `door_normal_threshold`   | `0.0`   | Min dot product for direction confirmation          |
| `polygon_padding_px`      | `0`     | Optional polygon padding (unused in v22 core)       |
| `minimum_visible_frames`  | `3`     | Min frames inside polygon before tracking           |

# Sack Counter — Change Log

## v20 (this release)

### Bug fixes

| # | Location | Description |
|---|----------|-------------|
| B-13 | `pipeline/state.py` | **Dead method removed.** `PipelineState._resolve()` was never called — `__getitem__` and `__setitem__` both inlined the same lookup logic. Removed to eliminate a maintenance trap (three places to update instead of one when changing `_FIELD_MAP`). |
| B-14 | `assignment.py` | **P#457-class bystander fix (function-boundary guard).** `assign_sacks_hungarian` now accepts a `persons_past_gate` set and excludes those IDs from the Hungarian cost matrix before solving. Previously the caller was solely responsible for pre-filtering; a missed filter silently let a bystander who had already been counted win assignment away from a legitimate carrier. |
| B-15 | `pipeline/ground_memory.py` | **GroundMemory stale-record leak.** `cleanup()` now prunes grounded records not seen for more than `stale_frames` (default 300, ≈15 s at 20 fps) frames. On long-running sessions, sacks whose tracker ID died while still on the floor accumulated indefinitely. |
| B-16 | `pipeline/ground_memory.py` | **Duplicate `_cosine` helper removed.** `DeepEmbedder.cosine` is the canonical implementation; the local copy in `ground_memory.py` was an identical duplicate. |

### Refactoring

| Location | Description |
|----------|-------------|
| `pipeline/state.py` | `AssignBugState` renamed → `CarrierGuardState`. The fields (`sack_carrier_stamp`, `just_evicted_sacks`, `orphaned_peaks`, `orphaned_sack_owner`) are now documented as first-class carrier management concepts with full field-level docstrings, not temporary workarounds. The `guards` group attribute name is unchanged so all existing call-sites continue to work. |

### Version strings
All module docstrings, log filenames, window titles, and output filenames
updated from v18/v19 → v20.

---

## v19 (previous)

* Direction-aware gate: `GateCounter` accepts `line_x`, `direction`, `offset`
  instead of two raw x-positions. `gate.A` / `gate.B` kept as legacy aliases.
* Interactive calibration screen (`calibration.py`) replaces CLI-only gate setup.
* `PipelineState` typed dataclass replaces flat 25-key dict (dict-like interface retained).
* Critical Bug #2 fix: LR-mode peak window was always False (offsets added in wrong direction).
* Critical Bug #3 fix: FIX-2 position-based exit buffer 5 px → 30 px.
* Bug #4–#12 fixes (see v19 `__init__.py` for full list).

## v18 (earlier)

* Modular `pipeline/` sub-package introduced.
* `SackStateMachineRegistry`, `AnalyticsEngine`, `GroundMemory`, `PipelineSession`.
* YAML config, structured logging, type hints throughout.

## v21 — Door-Zone + Disappearance Counting

Replaces the two-line gate crossing trigger with a polygon-zone + disappearance trigger.

### New files
- `sack_counter/door_polygon.py` — `DoorPolygon` geometry class + `calibrate_door_polygon()` interactive UI
- `sack_counter/pipeline/door_crossing.py` — `update_door_zone`, `process_door_disappearance`, `check_door_reentry`
- `tests/test_door_polygon.py` — geometry unit tests
- `tests/test_door_crossing.py` — crossing logic unit tests

### Deleted files
- `sack_counter/gate.py` — `GateCounter` removed entirely
- `sack_counter/calibration.py` — replaced by `door_polygon.calibrate_door_polygon()`

### Modified files
- `sack_counter/pipeline/state.py` — added `DoorState`, `person_position_history`
- `sack_counter/pipeline/counting.py` — removed `process_gate_crossings`, updated `update_peak_counts` to use door centroid
- `sack_counter/pipeline/orchestrator.py` — removed `GateCounter`, added `DoorPolygon` + `set_door()`
- `sack_counter/pipeline/person_tracker.py` — removed gate exit/re-entry logic, added position history
- `sack_counter/main.py` — wired door calibration and door crossing calls
- `sack_counter/box_pipeline.py` — removed gate parameter (box gate logic removed)
- `sack_counter/drawing.py` — removed `draw_gate`; `DoorPolygon.draw()` is self-contained
- `sack_counter/config.py` / `config.yaml` — new door-zone keys, gate keys removed
- `sack_counter/__init__.py` — exports `DoorPolygon` instead of `GateCounter`
- `run.py` — removed `--gate-x`, `--gate-gap`, `--direction` args
- `tests/conftest.py` — cv2 no longer stubbed (needed by DoorPolygon)
- `tests/test_counting.py` — gate tests replaced with door window direction tests
