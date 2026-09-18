"""HTTP error types."""
from __future__ import annotations


class HttpError(Exception):
    """Base error carrying a status code."""

    status = 500

    def __init__(self, message: str = "") -> None:
        super().__init__(message)
        self.message = message


class NotFound(HttpError):
    """404."""

    status = 404


class Unauthorized(HttpError):
    """401."""

    status = 401
