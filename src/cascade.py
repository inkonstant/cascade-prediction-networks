"""Simple data structures for representing cascades and their events.

This module defines a `Cascade` class which stores information about a single
information diffusion cascade.  It also defines a small helper function for
sorting events by time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class Event:
    """Represents a single retweet event in a cascade.

    Attributes
    ----------
    user: int
        The user who retweeted.
    parent: int
        The user from whom the message was retweeted.
    time: float
        Time (in seconds) since the original publish time when this retweet occurred.
    """

    user: int
    parent: int
    time: float


@dataclass
class Cascade:
    """Container for cascade information.

    A `Cascade` stores the ID of the message, the ID of the root user, the
    publish time, and a list of retweet events.  Each event knows its retweet
    time relative to the publish time and the parent user from whom the retweet
    happened.  The order of events is not guaranteed; call `sort_events()` to
    sort them chronologically.
    """

    cid: int
    root: int
    publish_time: float
    events: List[Event] = field(default_factory=list)

    def sort_events(self) -> None:
        """Sort events in-place by time ascending."""
        self.events.sort(key=lambda e: e.time)

    def add_event(self, user: int, parent: int, time: float) -> None:
        """Add a new event to the cascade.

        This method does **not** check for duplicates; callers should avoid
        adding multiple events for the same user unless they intend to keep
        all occurrences.
        """
        self.events.append(Event(user=user, parent=parent, time=time))

    def n_events(self) -> int:
        """Return the number of retweet events in this cascade."""
        return len(self.events)

    def get_k_prefix(self, k: int) -> Optional["Cascade"]:
        """Return a new `Cascade` containing only the first `k` events.

        The returned cascade has the same root and publish time and contains
        only the earliest `k` events sorted by time.  If the cascade has
        fewer than `k` events, `None` is returned.
        """
        self.sort_events()
        if len(self.events) < k:
            return None
        prefix = Cascade(cid=self.cid, root=self.root, publish_time=self.publish_time)
        prefix.events = self.events[:k].copy()
        return prefix

    def unique_users(self) -> List[int]:
        """Return the list of unique users in the cascade (including root)."""
        users = {self.root}
        users.update(event.user for event in self.events)
        return sorted(users)


def earliest_event_by_user(events: List[Event]) -> List[Event]:
    """Deduplicate events, keeping only the earliest event for each user.

    Parameters
    ----------
    events: list of `Event`
        The events parsed from a cascade.  Multiple events may have the same
        `.user` because a user can appear in several retweet paths.  Only the
        earliest event is kept for each user.

    Returns
    -------
    list of `Event`
        A new list of events containing at most one event per user, the
        earliest by time.
    """
    by_user: Dict[int, Event] = {}
    for ev in events:
        if ev.user not in by_user or ev.time < by_user[ev.user].time:
            by_user[ev.user] = ev
    return list(by_user.values())
