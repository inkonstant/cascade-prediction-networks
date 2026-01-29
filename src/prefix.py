"""Functions for generating k‑prefix cascades."""

from __future__ import annotations

from typing import List

from .cascade import Cascade


def generate_prefix_cascades(cascades: List[Cascade], k: int) -> List[Cascade]:
    """Generate k‑prefix cascades from a list of full cascades.

    Parameters
    ----------
    cascades: list of `Cascade`
        Full cascade objects (with all events).
    k: int
        The number of events to keep in each prefix.

    Returns
    -------
    list of `Cascade`
        Cascades where only the first `k` events are kept.  If a cascade has
        fewer than `k` events it is discarded.
    """
    prefixes: List[Cascade] = []
    for cascade in cascades:
        pref = cascade.get_k_prefix(k)
        if pref is not None:
            prefixes.append(pref)
    return prefixes
