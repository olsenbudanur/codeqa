"""Route handlers."""
from __future__ import annotations

from miniapp.api.errors import NotFound
from miniapp.auth.users import UserStore


class BaseRoute:
    """A route matches a path and handles a request dict."""

    path = "/"

    def handle(self, request: dict) -> dict:
        """Return a response dict."""
        raise NotImplementedError


class HealthRoute(BaseRoute):
    """Liveness check."""

    path = "/health"

    def handle(self, request: dict) -> dict:
        return {"status": 200, "body": "ok"}


class UsersRoute(BaseRoute):
    """Look up a user by id."""

    path = "/users"

    def __init__(self, store: UserStore) -> None:
        self.store = store

    def handle(self, request: dict) -> dict:
        user = self.store.get(request.get("query", {}).get("id", ""))
        if user is None:
            raise NotFound("no such user")
        return {"status": 200, "body": user.name}
