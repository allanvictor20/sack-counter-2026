"""
sack_counter.web — The console front end.

A local web application over the existing pipeline: the six screens of
the Sack Counter Console design, served by FastAPI, with the annotated
frames streamed into the Live view.

Nothing here reimplements counting.  Sessions run through the same
``main.run`` and ``run_exit_counter`` the CLI uses, via their frame-sink
hook, so the browser and the terminal show the same numbers.

    sack-counter --web          (or: python -m sack_counter.web)
"""

from .session import MANAGER, SessionManager, SessionOptions

__all__ = ["MANAGER", "SessionManager", "SessionOptions",
           "create_app", "serve"]


def create_app():
    """Return the FastAPI application (imported lazily — the CLI never needs it)."""
    from .app import app
    return app


def serve(host: str = "127.0.0.1", port: int = 8000,
          open_browser: bool = True, reload: bool = False) -> None:
    """
    Run the console.

    Binds to localhost by default: the app has no authentication and can
    start a session that reads a camera, so it should not be reachable
    from the network unless the operator says so explicitly.

    Args:
        host:         Interface to bind.
        port:         Port to listen on.
        open_browser: Open a browser window once the server is up.
        reload:       Auto-reload on code changes (development only).
    """
    import webbrowser

    from ..console import force_utf8_stdio
    from ..version import VERSION_TAG

    try:
        import uvicorn
    except ImportError as exc:                     # pragma: no cover
        raise SystemExit(
            "The web console needs its extra dependencies.\n"
            "  pip install 'sack-counter[web]'   (or: pip install "
            "fastapi uvicorn jinja2)"
        ) from exc

    force_utf8_stdio()
    url = f"http://{host}:{port}"
    print(f"\n  Sack Counter {VERSION_TAG} — web console")
    print(f"  {url}\n")
    if host not in ("127.0.0.1", "localhost"):
        print("  ! This console has no authentication and is bound to "
              f"{host}.\n    Anyone who can reach it can start a session.\n")
    if open_browser and not reload:
        webbrowser.open(url)

    uvicorn.run("sack_counter.web.app:app", host=host, port=port,
                reload=reload, log_level="info")
