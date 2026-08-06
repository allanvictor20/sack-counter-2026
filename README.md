# Sack Counter AI

An automated computer-vision pipeline for real-time detection, tracking,
door-crossing analysis, and exit monitoring of sacks passing through
checkpoints or doorways.

---

## Overview

**Sack Counter AI** monitors and counts sacks in warehouse and logistics
environments. It processes video to track workers, detect carried sacks,
define virtual landing zones and door polygons, and register crossing
events without double-counting.

Two independent modes:

| Mode | What it counts | Primary signal |
|---|---|---|
| `enter` | Sacks carried **into** a room | A sack stamped to a carrier crosses the door plane |
| `exit`  | Sacks leaving a room | A sack comes to rest inside the landing zone |

---

## Key features

- **Multi-object tracking** — separate person and sack trackers
  (`person_tracker.py`, `sack_tracker.py`).
- **Door polygon & landing zone** — click-to-calibrate geometry
  (`door_polygon.py`, `landing_zone.py`). Direction is decided by
  projection onto the door normal, so camera angle does not matter.
- **Crossing detection** — state-machine-based crossing logic
  (`door_crossing.py`, `exit_crossing.py`, `state_machine.py`).
- **Re-identification** — appearance embeddings (`embedder.py`) re-link a
  worker whose track is dropped, plus ground memory (`ground_memory.py`)
  to ignore sacks already resting on the floor.
- **Analytics & reporting** — per-worker stats, throughput, and JSON logs
  (`analytics.py`, `reporter.py`, `exit_reporter.py`).

---

## Installation

### Prerequisites

- **Python 3.10+** (tested on 3.12 and 3.14)
- A CUDA-capable GPU is optional; the pipeline runs on CPU.

### Setup

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

pip install -e .            # pipeline only
pip install -e ".[web]"     # plus the browser console
pip install -e ".[dev]"     # plus the test tooling
```

Installing gives you a `sack-counter` command that works from any
directory. `pip install -r requirements.txt` still works on a checkout
if you would rather not install the package.

### Model weights

The detector weights are **not** in this repository — they are large
binaries and are gitignored. You need:

| File | Used by | Classes |
|---|---|---|
| `best.pt` | enter mode (`model_path`) | `person`, `sack`, `box` |
| `exit_best.pt` | exit mode (`exit_model_path`) | `sack` |

Place them in the project root, or point `model_path` /
`exit_model_path` in `config.yaml` somewhere else.

Class indices are resolved **by name** from the model, so the ordering
inside your `.pt` does not matter — but the names must be recognisable
(`sack`/`bag`, `person`/`worker`, `box`/`carton`). If a name cannot be
matched, enter mode warns and falls back to `0=person, 1=sack, 2=box`;
exit mode refuses to run rather than count the wrong class.

Exit mode uses a separate model because the entry model was trained
mostly on *carried* sacks and scores ground-lying sacks poorly (median
confidence ~0.10). If `exit_model_path` is unset it falls back to
`model_path`.

---

## Usage

Two front ends over the same pipeline. The browser console is the
easier one; the CLI is what you want for scripted and headless runs.

```bash
sack-counter --web             # browser console at http://127.0.0.1:8000
sack-counter <source> [options]

# from a checkout, without installing:
python run.py --web
python run.py <source> [options]
```

`<source>` is a video file path, or the literal `webcam`.

| Option | Meaning |
|---|---|
| `--mode {enter,exit}` | Counting mode. Prompts interactively if omitted. |
| `--conf-sack FLOAT` | Override sack detection confidence. |
| `--conf-person FLOAT` | Override person confidence (enter mode). |
| `--conf-box FLOAT` | Override box confidence (enter mode). |
| `--save` | Write an annotated output video. |
| `--headless` | No display window; requires a saved calibration. |
| `--config PATH` | YAML or JSON config file. |
| `--use-door-crossing` | Exit mode: enable door crossing as a secondary cross-check. |
| `--web` | Serve the browser console instead of counting here. |
| `--host` / `--port` | Console bind address and port (default `127.0.0.1:8000`). |

Examples:

```bash
sack-counter video.mp4                      # prompts for mode
sack-counter video.mp4 --mode exit
sack-counter video.mp4 --mode enter --save
sack-counter webcam --mode exit
```

### Web console

```bash
sack-counter --web
```

Six screens over the same pipeline: **Live** (annotated frames, worker
meters, an activity feed), **Setup**, **Calibrate** (click the doorway
on a real frame), **Report**, **History**, and **Detection check**.

It binds to `127.0.0.1` deliberately. The console has no authentication
and can start a session that opens a camera, so it is local-only until
you pass `--host` explicitly — and it says so at startup when you do.

### Calibration

On the first interactive run you are asked to click the geometry:

- **Enter mode** — 4 door corners, then one point inside the room (which
  tells the system which side of the door is "in").
- **Exit mode** — the door polygon, then 3–8 points for the landing zone
  on the floor where sacks come to rest.

The result is saved to **`calibration.yaml`**, and merged automatically on
subsequent runs — so `--headless` works from then on. Delete that file to
re-calibrate. It is deliberately kept out of `config.yaml` (so your
comments there are never rewritten) and out of git (a polygon clicked on
one camera angle is meaningless on another).

### Configuration

`config.yaml` holds every tunable. `sack_counter/config.py` defines the
defaults and is the single source of truth — any key missing from your
file falls back to the value there.

> If a config file fails to parse, **all** settings revert to defaults.
> The loader prints a loud warning when this happens; do not trust counts
> from a run that printed it.

### Diagnostics

To check whether a detection problem is the model or the pipeline:

```bash
python diagnose_sacks.py video.mp4 --model best.pt --every 30
```

Runs raw YOLO with no pipeline logic at a low confidence floor, prints a
per-frame detection count and a confidence histogram, and writes
annotated sample frames to `diagnostic_frames/`.

---

## Output

| File | Contents |
|---|---|
| `delivery_log_v23_<timestamp>.json` | Enter mode: totals, per-delivery events, anomalies, analytics |
| `exit_log_v23_<timestamp>.json` | Exit mode: totals, exit events, discrepancy flags |
| `sack_counter.log` | Full debug trace (per-frame stamping, peaks, commits) |
| `output_v23.mp4` | Annotated video, with `--save` |

Both logs are timestamped, so a session never overwrites the one before
it and every run shows up in the console's **History** screen.

---

## Testing

```bash
pytest
```

The suite is GPU-free and needs no weights or footage. Unit tests drive
the pipeline on synthetic state; `tests/test_end_to_end.py` runs a whole
session — loop, report and log — against a generated video and a stub
detector, so the seams between stages are covered too.

`tests/test_regressions.py` pins the specific bugs found in past code
reviews; keep those passing. CI runs the suite and `ruff` on every push
and pull request.

---

## Project layout

```text
sack-counter-2026/
├── pyproject.toml            # Package metadata, extras, ruff + pytest config
├── config.yaml               # All tunables
├── calibration.yaml          # Written by calibration (gitignored)
├── run.py                    # From-checkout shortcut to sack_counter.cli
├── diagnose_sacks.py         # Raw-detection diagnostic
├── sack_counter/
│   ├── cli.py                # Command line (installed as `sack-counter`)
│   ├── main.py               # Enter-mode video loop
│   ├── detections.py         # Tracker output -> pipeline tuples
│   ├── session_view.py       # What a host sees of a running session
│   ├── session_log.py        # Timestamped log naming, both modes
│   ├── config.py             # Config loader + DEFAULT_CFG
│   ├── version.py            # Single source of truth for the version
│   ├── model_classes.py      # Resolve YOLO class indices by name
│   ├── console.py            # UTF-8 stdio helper
│   ├── door_polygon.py       # Door geometry + calibration
│   ├── embedder.py           # Appearance embeddings (ReID)
│   ├── trackers.py           # Motion, ReID, ownership, ghost sacks
│   ├── assignment.py         # Hungarian sack-to-person assignment
│   ├── theme.py              # Design tokens + OpenCV drawing primitives
│   ├── drawing.py            # The on-frame console overlay
│   ├── pipeline/             # Enter-mode per-frame stages
│   │   ├── orchestrator.py   # PipelineSession — owns all components
│   │   ├── state.py          # Typed state container
│   │   ├── field_view.py     # Flat dict access, shared by both containers
│   │   ├── geometry.py       # Where a person is, relative to the door
│   │   ├── frame.py          # One frame through every stage, in order
│   │   ├── person_tracker.py
│   │   ├── sack_tracker.py
│   │   ├── counting.py       # Peak counting + carrier stamping
│   │   ├── door_crossing.py  # Crossing commits + orphan rescue
│   │   ├── state_machine.py  # Per-sack lifecycle
│   │   ├── ground_memory.py
│   │   ├── analytics.py
│   │   └── reporter.py
│   ├── web/                  # Browser console (optional extra)
│   │   ├── app.py            # Routes
│   │   ├── session.py        # Runs a session in a worker thread
│   │   ├── history.py        # Reads past session logs
│   │   ├── diagnostics.py    # Detection check
│   │   ├── templates/
│   │   └── static/
│   └── exit/                 # Exit-mode sub-pipeline
│       ├── exit_main.py
│       ├── exit_tracker.py
│       ├── exit_crossing.py
│       ├── exit_landing.py
│       ├── landing_zone.py
│       └── exit_reporter.py
└── tests/
```
