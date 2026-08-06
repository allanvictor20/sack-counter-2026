"""
cli.py — The command line, importable rather than script-only.

This lived in ``run.py`` at the repository root, which meant the tool
only worked from a checkout and only if the current directory happened
to be the right one.  It is now a module with a ``main()`` entry point,
so ``pip install`` gives a real ``sack-counter`` command; ``run.py``
still works and simply calls in here.

The web console is reachable from the same place as everything else —
``sack-counter --web`` — instead of being an undocumented
``python -m sack_counter.web`` that nothing pointed at.
"""

from __future__ import annotations

import argparse
import sys

from .version import VERSION_TAG

_EPILOG = """
examples:
  sack-counter video.mp4                 count sacks coming in
  sack-counter video.mp4 --mode exit     count sacks going out
  sack-counter video.mp4 --save          also write an annotated video
  sack-counter webcam --mode enter       use the camera
  sack-counter --web                     open the browser console
"""


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, exposed so tests can exercise it directly."""
    parser = argparse.ArgumentParser(
        prog="sack-counter",
        description=f"Sack Counter {VERSION_TAG} — count sacks in or out",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source", nargs="?",
                        help="Video file path, or 'webcam'. "
                             "Not needed with --web.")
    parser.add_argument("--mode", choices=["enter", "exit"], default=None,
                        help="Counting mode. Prompted for if omitted.")
    parser.add_argument("--conf-sack", type=float, default=None,
                        help="Sack detection confidence threshold")
    parser.add_argument("--conf-person", type=float, default=None,
                        help="Person detection confidence (enter mode only)")
    parser.add_argument("--conf-box", type=float, default=None,
                        help="Box detection confidence (enter mode only)")
    parser.add_argument("--save", action="store_true",
                        help="Save an annotated output video")
    parser.add_argument("--headless", action="store_true",
                        help="Run with no display window (needs a saved "
                             "calibration)")
    parser.add_argument("--config", default=None,
                        help="Path to a YAML or JSON config file")
    parser.add_argument("--use-door-crossing", action="store_true",
                        help="(exit mode) Also count at the door, alongside "
                             "the landing zone")

    web = parser.add_argument_group("web console")
    web.add_argument("--web", action="store_true",
                     help="Serve the browser console instead of running a "
                          "session here")
    web.add_argument("--host", default="127.0.0.1",
                     help="Console bind address (default: 127.0.0.1). The "
                          "console has no authentication — only widen this "
                          "on a network you trust.")
    web.add_argument("--port", type=int, default=8000,
                     help="Console port (default: 8000)")
    return parser


def prompt_mode() -> str:
    """Ask which direction to count when ``--mode`` was not given."""
    print("\n" + "=" * 44)
    print(f"  SACK COUNTER {VERSION_TAG}")
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


def main(argv: list[str] | None = None) -> int:
    """
    Run the CLI.

    Args:
        argv: Arguments to parse.  Defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.web:
        from .web import serve
        serve(host=args.host, port=args.port)
        return 0

    if not args.source:
        parser.error("a video source is required (or pass --web for the "
                     "browser console)")

    mode = args.mode or prompt_mode()

    if mode == "exit":
        from .exit import run_exit_counter
        run_exit_counter(
            source            = args.source,
            conf_sack         = args.conf_sack,
            save_output       = args.save,
            headless          = args.headless,
            config_path       = args.config,
            use_door_crossing = args.use_door_crossing,
        )
    else:
        from .main import run
        run(
            source      = args.source,
            conf_sack   = args.conf_sack,
            conf_person = args.conf_person,
            conf_box    = args.conf_box,
            save_output = args.save,
            headless    = args.headless,
            config_path = args.config,
        )
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
