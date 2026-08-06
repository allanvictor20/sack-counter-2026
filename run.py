#!/usr/bin/env python3
"""
run.py — Command-line entry point for Sack Counter.

The implementation lives in :mod:`sack_counter.cli` so that it also
works when the package is installed (``sack-counter …``) rather than
only from a checkout.  This script is the from-source shortcut.

Usage
-----
    python run.py <source> [options]
    python run.py --web

Examples
--------
    python run.py video.mp4
    python run.py video.mp4 --mode exit
    python run.py video.mp4 --mode enter --save
    python run.py video.mp4 --headless --config config.yaml
    python run.py webcam --mode exit
    python run.py --web                 # browser console
"""

from sack_counter.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
