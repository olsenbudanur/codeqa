"""Configuration defaults and loading."""
from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_TIMEOUT = 30
DEFAULT_SECRET = "change-me"
MAX_SESSIONS = 1000


@dataclass
class Settings:
    """Runtime settings for the service."""

    secret: str = DEFAULT_SECRET
    timeout: int = DEFAULT_TIMEOUT
    debug: bool = False


def load_settings() -> Settings:
    """Build Settings from environment variables, falling back to defaults."""
    return Settings(
        secret=os.environ.get("MINIAPP_SECRET", DEFAULT_SECRET),
        timeout=int(os.environ.get("MINIAPP_TIMEOUT", DEFAULT_TIMEOUT)),
        debug=os.environ.get("MINIAPP_DEBUG") == "1",
    )
