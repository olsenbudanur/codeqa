"""Application object wiring middleware and routes."""
from __future__ import annotations

from miniapp.api.errors import HttpError, NotFound
from miniapp.api.middleware import auth_middleware, logging_middleware
from miniapp.api.routes import BaseRoute, HealthRoute, UsersRoute
from miniapp.auth.users import UserStore
from miniapp.config import Settings, load_settings


class App:
    """Dispatches requests to routes after running middleware."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.routes: dict[str, BaseRoute] = {}

    def add_route(self, route: BaseRoute) -> None:
        """Register a route by its path."""
        self.routes[route.path] = route

    def dispatch(self, request: dict) -> dict:
        """Run middleware, then the matching route; convert HttpError to a response."""
        try:
            request = logging_middleware(request)
            if request.get("path") != "/health":
                request = auth_middleware(request, self.settings.secret)
            route = self.routes.get(request.get("path", "/"))
            if route is None:
                raise NotFound(request.get("path", "/"))
            return route.handle(request)
        except HttpError as exc:
            return {"status": exc.status, "body": exc.message}


def create_app(settings: Settings | None = None) -> App:
    """Build an App with the default routes."""
    app = App(settings or load_settings())
    store = UserStore()
    app.add_route(HealthRoute())
    app.add_route(UsersRoute(store))
    return app
