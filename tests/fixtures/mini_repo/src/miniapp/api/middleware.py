"""Request middleware."""
from __future__ import annotations

from miniapp.api.errors import Unauthorized
from miniapp.auth.session import SessionExpired, validate_session
from miniapp.auth.tokens import TokenError


def auth_middleware(request: dict, secret: str) -> dict:
    """Validate the session token on a request; map auth failures to 401."""
    token = request.get("headers", {}).get("authorization", "")
    try:
        session = validate_session(token, secret)
    except SessionExpired as exc:
        raise Unauthorized(f"expired: {exc}") from exc
    except TokenError as exc:
        raise Unauthorized(f"invalid: {exc}") from exc
    request["session"] = session
    return request


def logging_middleware(request: dict) -> dict:
    """Record the request path."""
    request.setdefault("log", []).append(request.get("path", "/"))
    return request
