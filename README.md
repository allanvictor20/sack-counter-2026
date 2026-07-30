# Sack Counter v18

Real-time computer vision system for counting sacks and boxes delivered per person,
using YOLOv8 detection, deep ReID tracking, Hungarian algorithm assignment, and a
two-line gate counter.

## What's new in v18

### Architecture overhaul
The original 1,215-line `main.py` has been refactored into a clean `pipeline/` sub-package:

```
sack_counter/
├── pipeline/
│   ├── state_machine.py   # Per-sack lifecycle: DETECTED → CARRIED → DELIVERED
│   ├── analytics.py       # Per-worker stats, throughput, minute-by-minute trends
│   ├── ground_memory.py   # Memory of sacks already on the floor
│   └── orchestrator.py    # PipelineSession — owns all components
├── main.py                # Loop only (~350 lines, down from 1,215)
├── config.py              # YAML + JSON config loading
├── logger.py              # Structured logging (DEBUG/INFO/WARNING/ERROR)
└── ...                    # (all v17 modules unchanged)
```

### Sack state machine
Each tracked sack now has an explicit lifecycle:
```
ON_GROUND → DETECTED → CONFIRMED → PICKED_UP → CARRIED → APPROACHING → DELIVERED
```
Invalid transitions are rejected, making behaviour predictable and debuggable.

### Ground-object memory (`GroundMemory`)
Sacks already resting on the floor are now remembered across frames.
A pickup is only confirmed after `lift_owner_stable_frames` consecutive moving frames,
preventing a passing carrier from accidentally triggering a floor sack count.

### Analytics engine (`AnalyticsEngine`)
Every delivery session now produces:
- Deliveries per worker
- Average load size per carrier
- Peak throughput (items / minute)
- Minute-by-minute trend buckets
- Ownership confidence statistics

These are written to `delivery_log_v18.json` under the `"analytics"` key.

### YAML configuration
```yaml
# config.yaml (auto-loaded from current directory)
conf_sack: 0.35
peak_window_px: 350
gate_gap_px: 40
```
Run `python run.py video.mp4` — `config.yaml` is picked up automatically.
Pass `--config path/to/other.yaml` to override.

### Structured logging
```
INFO    — deliveries, gate crossings, re-entry events
WARNING — low ownership confidence anomalies
DEBUG   — frame-by-frame assignment / stamp / peak details (file only)
```
Console shows WARNING+ only; full detail goes to `sack_counter_v18.log`.

### Type hints and docstrings
Every public function and class now has full type annotations and docstrings.

### Unit tests
```
tests/
├── test_assignment.py     # association_score, carry_zone_bounds, hungarian
├── test_confidence.py     # ConfidenceTracker, confidence_class
├── test_state_machine.py  # SackStateMachineRegistry transitions
├── test_analytics.py      # AnalyticsEngine per-worker stats
├── test_ground_memory.py  # GroundMemory pickup detection
└── test_counting.py       # GateCounter, config loader
```

Run all tests:
```bash
python -m pytest tests/ -v
# or without pytest:
python -m unittest discover tests/
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Basic
python run.py video.mp4

# With YAML config
python run.py video.mp4 --config config.yaml

# Save output video, no display
python run.py video.mp4 --save --headless

# Override gate position and confidence
python run.py video.mp4 --gate-x 640 --conf-sack 0.40
```

## Project structure

```
sack_counter_v18/
├── config.yaml              ← tune parameters without touching code
├── requirements.txt
├── run.py                   ← CLI entry point
├── sack_counter/
│   ├── __init__.py
│   ├── main.py              ← processing loop (~350 lines)
│   ├── config.py            ← YAML/JSON loader
│   ├── logger.py            ← structured logging
│   ├── assignment.py        ← Hungarian assignment
│   ├── confidence.py        ← delivery confidence scoring
│   ├── trackers.py          ← motion/ReID/ownership/ghost trackers
│   ├── gate.py              ← two-line gate counter
│   ├── embedder.py          ← MobileNetV3 deep embedder
│   ├── drawing.py           ← OpenCV overlay rendering
│   ├── box_pipeline.py      ← box counting sub-pipeline
│   ├── colors.py            ← colour palette
│   └── pipeline/
│       ├── __init__.py
│       ├── state_machine.py
│       ├── analytics.py
│       ├── ground_memory.py
│       └── orchestrator.py
└── tests/
    ├── test_assignment.py
    ├── test_confidence.py
    ├── test_state_machine.py
    ├── test_analytics.py
    ├── test_ground_memory.py
    └── test_counting.py
```

## All v17 bug fixes retained

- FIX-UNICODE: UTF-8 stdout/stderr on Windows
- FIX-REENTRY: persons_past_gate cleared on re-entry
- FIX-STAMP: sack_carrier_stamp cleared on re-entry
- FIX-TTL: orphaned-peak TTL = miss_frames
- FIX-PEAK0: explicit INFO log for every peak=0 crossing
- FIX 1–3b: ownership eviction on timeout / position / gate crossing
- BUG1-A/B: just_evicted_sacks / sack_carrier_stamp guards
- BUG2: orphaned_peaks registry for late-crossing sacks
