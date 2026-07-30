#!/usr/bin/env python3
"""
run.py — Command-line entry point for Sack Counter v23.

Modes
-----
  1) Count sacks ENTERING the room  (original v22 pipeline)
  2) Count sacks LEAVING  the room  (new exit pipeline)

Usage
-----
    python run.py <source> [options]

Examples
--------
    python run.py video.mp4
    python run.py video.mp4 --mode exit
    python run.py video.mp4 --mode enter --save
    python run.py video.mp4 --headless --config config.yaml
    python run.py webcam --mode exit
"""

import argparse
import sys


def _prompt_mode() -> str:
    """Interactively ask the user which mode to run."""
    print("\n" + "=" * 44)
    print("  SACK COUNTER v23")
    print("=" * 44)
    print("  [1]  Count sacks ENTERING the room")
    print("  [2]  Count sacks LEAVING  the room")
    print("  [q]  Quit")
    print("-" * 44)

    while True:
        choice = input("  Select mode [1/2/q]: ").strip().lower()
        if choice == "1":
            return "enter"
        if choice == "2":
            return "exit"
        if choice in ("q", "quit"):
            print("  Bye.")
            sys.exit(0)
        print("  Please enter 1, 2, or q.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Sack Counter v23 — enter or exit mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("source",
                    help="Video file path, or 'webcam'")
    ap.add_argument("--mode", choices=["enter", "exit"], default=None,
                    help="Counting mode: 'enter' (default) or 'exit'. "
                         "If omitted, you will be prompted.")
    ap.add_argument("--conf-sack",   type=float, default=None,
                    help="Sack detection confidence threshold")
    ap.add_argument("--conf-person", type=float, default=None,
                    help="Person detection confidence (enter mode only)")
    ap.add_argument("--conf-box",    type=float, default=None,
                    help="Box detection confidence (enter mode only)")
    ap.add_argument("--save",        action="store_true",
                    help="Save annotated output video")
    ap.add_argument("--headless",    action="store_true",
                    help="Run without display window (requires calibrated config)")
    ap.add_argument("--config",      default=None,
                    help="Path to YAML or JSON config file")
    ap.add_argument("--use-door-crossing", action="store_true",
                    help="(exit mode only) Enable door-crossing as a "
                         "secondary counting signal alongside the landing "
                         "zone. OFF by default — the landing zone alone "
                         "is the primary, more reliable counter.")

    args = ap.parse_args()

    # Determine mode
    mode = args.mode
    if mode is None:
        mode = _prompt_mode()

    if mode == "exit":
        from sack_counter.exit import run_exit_counter
        run_exit_counter(
            source            = args.source,
            conf_sack         = args.conf_sack,
            save_output       = args.save,
            headless          = args.headless,
            config_path       = args.config,
            use_door_crossing = args.use_door_crossing,
        )
    else:
        # Original enter-mode pipeline
        from sack_counter.main import run
        run(
            source      = args.source,
            conf_sack   = args.conf_sack,
            conf_person = args.conf_person,
            conf_box    = args.conf_box,
            save_output = args.save,
            headless    = args.headless,
            config_path = args.config,
        )
