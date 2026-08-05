"""
__main__.py — ``python -m sack_counter.web``

Starts the console on localhost.  Binds to 127.0.0.1 by default: the app
has no authentication and can start a session that reads a camera, so it
should not be reachable from the network unless you say so explicitly.
"""

from __future__ import annotations

import argparse
import webbrowser

from ..console import force_utf8_stdio
from ..version import VERSION_TAG


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

    force_utf8_stdio()
    url = f"http://{args.host}:{args.port}"
    print(f"\n  Sack Counter {VERSION_TAG} — web console")
    print(f"  {url}\n")
    if not args.no_browser and not args.reload:
        webbrowser.open(url)

    import uvicorn
    uvicorn.run("sack_counter.web.app:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
