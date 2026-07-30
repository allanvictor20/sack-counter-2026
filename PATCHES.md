# v19 Patch Instructions

These are the copy-paste changes for every file that needs **minor edits only**.
Each section tells you exactly which lines to find and what to replace them with.

---

## 1. `sack_counter/main.py`

### Change 1 — Add calibration import (top of imports, after `from .config import load_config`)

**Find:**
```python
from .config import load_config
```
**Replace with:**
```python
from .config import load_config
from .calibration import calibrate
```

---

### Change 2 — Update `run()` signature (remove gate_x / gate_gap, add headless guard)

**Find:**
```python
def run(
    source:      str,
    conf_sack:   float | None = None,
    conf_person: float | None = None,
    conf_box:    float | None = None,
    gate_x:      int | None   = None,
    gate_gap:    int | None   = None,
    save_output: bool         = False,
    headless:    bool         = False,
    config_path: str | None   = None,
) -> None:
```
**Replace with:**
```python
def run(
    source:      str,
    conf_sack:   float | None = None,
    conf_person: float | None = None,
    conf_box:    float | None = None,
    gate_x:      int | None   = None,   # headless/CLI fallback only
    gate_gap:    int | None   = None,   # headless/CLI fallback only
    direction:   str | None   = None,   # headless/CLI fallback only
    save_output: bool         = False,
    headless:    bool         = False,
    config_path: str | None   = None,
) -> None:
```

---

### Change 3 — Run calibration before building the session (replaces the old session-creation block)

**Find:**
```python
    # ── Build session (owns all stateful components) ──────────
    session = PipelineSession(
        cfg=cfg, frame_width=fw, frame_height=fh,
        src_fps=src_fps, gate_x=gate_x, gate_gap=gate_gap,
    )
```
**Replace with:**
```python
    # ── Calibration (interactive) or headless fallback ────────
    if headless:
        # CLI / batch mode: use gate_x, gate_gap, direction args
        calibration = None          # orchestrator uses gate_x / gate_gap
        if direction is not None:
            cfg["direction"] = direction
    else:
        offset = gate_gap if gate_gap is not None else cfg["gate_gap_px"]
        calibration = calibrate(cap, offset=offset)

    # ── Build session (owns all stateful components) ──────────
    session = PipelineSession(
        cfg=cfg, frame_width=fw, frame_height=fh,
        src_fps=src_fps,
        calibration=calibration,
        gate_x=gate_x,
        gate_gap=gate_gap,
    )
```

---

### Change 4 — Update the startup print for gate lines

**Find:**
```python
    print(f"  Gate A=x{gate.A}  Gate B=x{gate.B}\n")
```
**Replace with:**
```python
    print(f"  Direction    : {gate.direction}")
    print(f"  Entry line   : x={gate.entry_line}")
    print(f"  Exit  line   : x={gate.exit_line}\n")
```

---

### Change 5 — Peak state cleanup loop (uses gate.B → gate.exit_line)

**Find:**
```python
                if pid_cx is not None and pid_cx < gate.B - cfg["peak_freeze_px"] * 2:
```
**Replace with:**
```python
                if pid_cx is not None and pid_cx < gate.exit_line - cfg["peak_freeze_px"] * 2:
```

---

### Change 6 — State machine approach window (uses gate.A → gate.entry_line)

**Find:**
```python
        approach_open = gate.A + cfg["peak_window_px"]
```
**Replace with:**
```python
        approach_open = gate.entry_line + cfg["peak_window_px"]
```

---

## 2. `sack_counter/pipeline/counting.py`

### Change 1 — `update_peak_counts`: replace gate.A / gate.B references

**Find:**
```python
    approach_open = gate.A + cfg["peak_window_px"]
    freeze_edge   = gate.B + cfg["peak_freeze_px"]
```
**Replace with:**
```python
    approach_open = gate.entry_line + cfg["peak_window_px"]
    freeze_edge   = gate.exit_line  + cfg["peak_freeze_px"]
```

---

### Change 2 — `process_gate_crossings`: orphan guard uses gate.A

**Find:**
```python
            if sack_cx > gate.A:
```
**Replace with:**
```python
            if sack_cx > gate.entry_line:
```

---

### Change 3 — `_evict_residual_ownership`: fix3b zone uses gate.A (two occurrences on adjacent lines)

**Find:**
```python
        if state[\"sack_positions\"].get(
            s, (gate.A + fix3b_zone + 1, 0)
        )[0] >= gate.A - fix3b_zone
```
**Replace with:**
```python
        if state["sack_positions"].get(
            s, (gate.entry_line + fix3b_zone + 1, 0)
        )[0] >= gate.entry_line - fix3b_zone
```

---

## 3. `sack_counter/pipeline/person_tracker.py`

### Change 1 — FIX-REENTRY guard uses gate.A

**Find:**
```python
        if can_pid in state["persons_past_gate"] and cx > gate.A + 100:
            state["persons_past_gate"].discard(can_pid)
            logger.info(
                "P#%d re-entered scene at cx=%d (gate.A=%d) — "
                "removed from past_gate  frame=%d",
                can_pid, cx, gate.A, fn,
```
**Replace with:**
```python
        if can_pid in state["persons_past_gate"] and cx > gate.entry_line + 100:
            state["persons_past_gate"].discard(can_pid)
            logger.info(
                "P#%d re-entered scene at cx=%d (gate.entry_line=%d) — "
                "removed from past_gate  frame=%d",
                can_pid, cx, gate.entry_line, fn,
```

---

### Change 2 — FIX 2 position-based exit uses gate.B

**Find:**
```python
        if cx < gate.B - 5 and can_pid not in state["persons_past_gate"]:
```
**Replace with:**
```python
        if cx < gate.exit_line - 5 and can_pid not in state["persons_past_gate"]:
```

---

## 4. `sack_counter/drawing.py`

### Change 1 — `draw_gate`: rename labels and add direction arrow

**Find:**
```python
def draw_gate(frame, gate: GateCounter, fh, total_sacks, total_boxes):
    cv2.line(frame, (gate.A, 0), (gate.A, fh), C_LINE_A, 2)
    # Smaller gate labels
    cv2.putText(frame, "GATE A", (gate.A - 55, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_LINE_A, 1)
    cv2.line(frame, (gate.B, 0), (gate.B, fh), C_LINE_B, 1)
    cv2.putText(frame, "GATE B", (gate.B - 55, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_LINE_B, 1)
    # Gate total count kept larger — this is the key number
    cv2.putText(frame, f"Sacks: {total_sacks}  Boxes: {total_boxes}",
                (min(gate.A, gate.B) - 10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_LINE_A, 2)
```
**Replace with:**
```python
def draw_gate(frame, gate: GateCounter, fh, total_sacks, total_boxes):
    # Entry line (amber)
    cv2.line(frame, (gate.entry_line, 0), (gate.entry_line, fh), C_LINE_A, 2)
    cv2.putText(frame, "ENTRY", (gate.entry_line - 55, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_LINE_A, 1)
    # Exit line (red)
    cv2.line(frame, (gate.exit_line, 0), (gate.exit_line, fh), C_LINE_B, 1)
    cv2.putText(frame, "EXIT", (gate.exit_line - 55, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_LINE_B, 1)
    # Direction arrow between the two lines
    mid_y = fh // 2
    if gate.direction == "RL":
        arrow_p1 = (gate.entry_line + 30, mid_y)
        arrow_p2 = (gate.exit_line  - 30, mid_y)
    else:
        arrow_p1 = (gate.exit_line  + 30, mid_y)
        arrow_p2 = (gate.entry_line - 30, mid_y)
    cv2.arrowedLine(frame, arrow_p1, arrow_p2, (255, 255, 255), 2, tipLength=0.25)
    # Gate total count
    cv2.putText(frame, f"Sacks: {total_sacks}  Boxes: {total_boxes}",
                (min(gate.entry_line, gate.exit_line) - 10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_LINE_A, 2)
```

---

## 5. `config.py` — Add `direction` default

**Find:**
```python
    "gate_gap_px":            40,
    "gate_require_both":      True,
```
**Replace with:**
```python
    "gate_gap_px":            40,
    "gate_require_both":      True,
    "direction":              "RL",   # "RL" = right-to-left, "LR" = left-to-right
```

---

## 6. `config.yaml` — Add direction key

Add this line anywhere in the gate section (near `gate_gap_px`):

```yaml
direction:        RL      # RL = right-to-left  |  LR = left-to-right
```

---

## 7. `run.py` — Add `--direction` CLI argument

**Find:**
```python
    ap.add_argument("--gate-gap",     type=int,   default=None,
                    help="Pixel gap between gate lines A and B (default: from config)")
```
**Replace with:**
```python
    ap.add_argument("--gate-gap",     type=int,   default=None,
                    help="Pixel gap between gate lines A and B (default: from config)")
    ap.add_argument("--direction",    type=str,   default=None,
                    choices=["RL", "LR"],
                    help="Counting direction: RL=right-to-left, LR=left-to-right "
                         "(headless only; calibration screen sets this interactively)")
```

Also update the `run(...)` call at the bottom of `run.py`:

**Find:**
```python
    run(
        source      = args.source,
        conf_sack   = args.conf_sack,
        conf_person = args.conf_person,
        conf_box    = args.conf_box,
        gate_x      = args.gate_x,
        gate_gap    = args.gate_gap,
        save_output = args.save,
        headless    = args.headless,
        config_path = args.config,
    )
```
**Replace with:**
```python
    run(
        source      = args.source,
        conf_sack   = args.conf_sack,
        conf_person = args.conf_person,
        conf_box    = args.conf_box,
        gate_x      = args.gate_x,
        gate_gap    = args.gate_gap,
        direction   = args.direction,
        save_output = args.save,
        headless    = args.headless,
        config_path = args.config,
    )
```
