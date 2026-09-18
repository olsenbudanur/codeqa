"""HMAC token signing and verification."""
from __future__ import annotations

import hashlib
import hmac


class TokenError(ValueError):
    """Raised when a token is malformed or its signature does not match."""


def sign_token(payload: str, secret: str) -> str:
    """Return payload joined with its hex HMAC-SHA256 signature."""
    sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def verify_token(token: str, secret: str) -> str:
    """Return the payload if the signature is valid, else raise TokenError."""
    if "." not in token:
        raise TokenError("malformed token")
    payload, sig = token.rsplit(".", 1)
    expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        raise TokenError("bad signature")
    return payload
