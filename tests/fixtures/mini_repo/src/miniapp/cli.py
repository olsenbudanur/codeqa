"""Command line entry point."""
from __future__ import annotations

import sys

from miniapp.api.app import create_app


def main(argv: list[str] | None = None) -> int:
    """Dispatch one request given as a path argument and print the body."""
    argv = argv if argv is not None else sys.argv[1:]
    app = create_app()
    response = app.dispatch({"path": argv[0] if argv else "/health"})
    print(response["body"])
    return 0 if response["status"] == 200 else 1
