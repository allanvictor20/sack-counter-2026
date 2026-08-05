"""
sack_counter.web — The console front end.

A local web application over the existing pipeline: the six screens of
the Sack Counter Console design, served by FastAPI, with the annotated
frames streamed into the Live view.

Nothing here reimplements counting.  Sessions run through the same
``main.run`` and ``run_exit_counter`` the CLI uses, via their frame-sink
hook, so the browser and the terminal show the same numbers.

    python -m sack_counter.web
"""

from .session import MANAGER, SessionManager, SessionOptions

__all__ = ["MANAGER", "SessionManager", "SessionOptions", "create_app"]


def create_app():
    """Return the FastAPI application (imported lazily — the CLI never needs it)."""
    from .app import app
    return app
