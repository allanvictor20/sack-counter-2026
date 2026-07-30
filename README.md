

```markdown
# Sack Counter AI

An automated Computer Vision pipeline for real-time detection, tracking, door-crossing analysis, and exit monitoring of sacks/bags passing through checkpoints or doorways.

---

## Overview

**Sack Counter AI** provides a high-assurance tracking and analytics system designed to monitor and count sacks in warehouse or logistics environments. The system processes video streams to track workers, detect carried or transported sacks, define virtual landing zones and door polygons, and accurately register crossing events to prevent double-counting.

---

## Key Features

- **Multi-Object Tracking & Pipeline Orchestration:** Separate, specialized tracking modules for persons and sacks (`person_tracker.py`, `sack_tracker.py`).
- **Door Polygon & Landing Zone Analysis:** Custom geometry configuration (`door_polygon.py`, `landing_zone.py`) to monitor specific checkpoint boundaries and entry/exit vectors.
- **Crossing & Exit Detection:** State-machine-based crossing logic (`door_crossing.py`, `exit_crossing.py`, `state_machine.py`) ensuring reliable event registration.
- **Ground Memory & Re-identification:** Built-in ground memory tracking (`ground_memory.py`) and feature embedding (`embedder.py`) to reduce false positive counts.
- **Analytics & Diagnostic Reporting:** Real-time logging (`logger.py`), exit reporting (`exit_reporter.py`), and diagnostic tooling (`diagnose_sacks.py`).

---

## Directory Structure

```text
sack-counter-2026/
├── config.yaml               # System parameters, threshold, and camera configs
├── requirements.txt          # Python dependencies
├── run.py                    # Application entry point
├── diagnose_sacks.py         # Utility script for diagnostics and debugging
├── sack_counter/
│   ├── main.py               # Main pipeline execution script
│   ├── config.py             # Configuration loader
│   ├── embedder.py           # Feature embeddings generator
│   ├── trackers.py           # Core tracking abstractions
│   ├── pipeline/             # Pipeline components
│   │   ├── orchestrator.py   # Main pipeline orchestrator
│   │   ├── person_tracker.py # Person detection and tracking
│   │   ├── sack_tracker.py   # Sack detection and tracking
│   │   ├── door_crossing.py  # Door line/polygon crossing detection
│   │   ├── counting.py       # Tallying and counting logic
│   │   ├── state_machine.py  # Pipeline state transitions
│   │   ├── ground_memory.py # Spatial memory tracking
│   │   └── analytics.py      # Metric gathering and logging
│   └── exit/                 # Dedicated exit sub-pipeline
│       ├── exit_main.py      # Exit monitoring runner
│       ├── exit_tracker.py   # Exit zone tracking
│       ├── exit_crossing.py  # Exit boundary crossing detection
│       └── landing_zone.py   # Landing zone spatial logic
└── tests/                    # Pytest test suite
    ├── exit/                 # Exit pipeline tests
    └── test_*.py             # Unit and integration tests

```

---

## Installation

### 1. Prerequisites

* **Python 3.12+**
* `pip` package manager
* (Optional) CUDA-enabled GPU for accelerated inference

### 2. Setup Virtual Environment

```bash
python -m venv venv

# On Linux / macOS:
source venv/bin/activate

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

---

## Usage

### Configuration

Adjust parameters such as confidence thresholds, model paths, door polygon coordinates, and video input paths in `config.yaml`.

### Running the Main Counter

To execute the main counting pipeline:

```bash
python run.py

```

Or run directly via the package:

```bash
python -m sack_counter.main

```

### Running Diagnostics

If you need to analyze video clips or troubleshoot detection issues:

```bash
python diagnose_sacks.py --input path/to/video.mp4

```

---

## Testing

The project includes a comprehensive test suite built with `pytest` covering pipeline states, assignment logic, door crossing, and exit tracking.

Run the test suite with:

```bash
pytest

```

To run test coverage on specific modules:

```bash
pytest tests/exit/

```

```

```
