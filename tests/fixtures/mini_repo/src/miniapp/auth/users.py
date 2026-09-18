"""In-memory user registry."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class User:
    """A registered user."""

    user_id: str
    name: str
    is_admin: bool = False


class UserStore:
    """Holds users by id."""

    def __init__(self) -> None:
        self._users: dict[str, User] = {}

    def add(self, user: User) -> None:
        """Register a user, replacing any existing entry."""
        self._users[user.user_id] = user

    def get(self, user_id: str) -> User | None:
        """Return the user or None."""
        return self._users.get(user_id)
