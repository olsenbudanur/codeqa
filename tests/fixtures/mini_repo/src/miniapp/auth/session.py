"""Session creation, validation and expiry."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from miniapp.auth.tokens import TokenError, sign_token, verify_token
from miniapp.config import DEFAULT_TIMEOUT


class SessionExpired(Exception):
    """Raised when a session token is past its expiry."""


@dataclass
class Session:
    """A signed, time-limited session."""

    user_id: str
    expires_at: float
    token: str = field(default="")

    def is_expired(self, now: float | None = None) -> bool:
        """True when the session is past expires_at."""
        return (now or time.time()) > self.expires_at


def create_session(user_id: str, secret: str, ttl: int = DEFAULT_TIMEOUT) -> Session:
    """Create a session whose token encodes the user id and expiry."""
    expires_at = time.time() + ttl
    token = sign_token(f"{user_id}:{expires_at}", secret)
    return Session(user_id=user_id, expires_at=expires_at, token=token)


def validate_session(token: str, secret: str, now: float | None = None) -> Session:
    """Verify the token signature and expiry; raise SessionExpired or TokenError."""
    payload = verify_token(token, secret)
    user_id, expires = payload.split(":", 1)
    session = Session(user_id=user_id, expires_at=float(expires), token=token)
    if session.is_expired(now):
        raise SessionExpired(f"session for {user_id} expired")
    return session


def expire_session(session: Session) -> Session:
    """Force a session to expire immediately."""
    session.expires_at = 0.0
    return session
