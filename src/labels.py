"""Functions for constructing labels for cascade prefixes."""

from __future__ import annotations

from typing import Dict, List

from .cascade import Cascade


def construct_labels(prefixes: List[Cascade], full_sizes: Dict[int, int], k: int) -> List[int]:
    """Create binary labels for prefix cascades based on the doubling rule.

    Parameters
    ----------
    prefixes: list of `Cascade`
        A list of k‑prefix cascades.  Each prefix must have exactly `k` events.
    full_sizes: dict[int, int]
        A mapping from cascade ID to the total number of retweet events in
        the full cascade.
    k: int
        The value of k used to generate the prefixes.

    Returns
    -------
    list of int
        Labels where 1 indicates that the full cascade size is at least `2*k`
        (i.e. the cascade will at least double in size relative to the prefix),
        and 0 otherwise.
    """
    labels: List[int] = []
    threshold = 2 * k
    for pref in prefixes:
        cid = pref.cid
        final_size = full_sizes.get(cid, 0)
        labels.append(1 if final_size >= threshold else 0)
    return labels
