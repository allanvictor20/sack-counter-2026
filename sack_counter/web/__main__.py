"""
__main__.py — ``python -m sack_counter.web``

Kept as an alias for ``sack-counter --web``; both land in
:func:`sack_counter.web.serve`, so the bind-address warning and the
missing-dependency message are the same whichever way it is started.
"""

from __future__ import annotations

import argparse

from ..version import VERSION_TAG
from . import serve


def main() -> None:
    ap = argparse.ArgumentParser(
        description=f"Sack Counter {VERSION_TAG} — web console")
    ap.add_argument("--host", default="127.0.0.1",
                    help="Interface to bind (default 127.0.0.1, local only)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true",
                    help="Auto-reload on code changes (development)")
    ap.add_argument("--no-browser", action="store_true",
                    help="Do not open a browser window on start")
    args = ap.parse_args()

    serve(host=args.host, port=args.port,
          open_browser=not args.no_browser, reload=args.reload)


if __name__ == "__main__":
    main()
