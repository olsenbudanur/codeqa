"""Dict-backed key/value store."""
from __future__ import annotations


class MemoryStore:
    """Simple in-process store."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def put(self, key: str, value: str) -> None:
        """Store a value."""
        self._data[key] = value

    def get(self, key: str, default: str | None = None) -> str | None:
        """Fetch a value."""
        return self._data.get(key, default)

    def __len__(self) -> int:
        return len(self._data)
