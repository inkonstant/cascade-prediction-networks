"""Feature extraction for cascade prefixes."""

from __future__ import annotations

from typing import Dict
import numpy as np

from .cascade import Cascade
from .build_tree import build_tree, structural_metrics


def temporal_features(prefix: Cascade) -> Dict[str, float]:
    """Compute temporal features from a prefix cascade.

    The events in the prefix should already be sorted by time.  The
    following features are extracted:

    - `time_to_k`: time of the k‑th retweet (last event).
    - `mean_inter_time`: average inter‑retweet time.
    - `var_inter_time`: variance of inter‑retweet times.
    - `half_life_ratio`: ratio of time to half of the retweets over `time_to_k`.
    - `speed_change`: ratio of the average inter‑time in the first half to the second half.

    If a prefix has fewer than two events some values will be zero.
    """
    evs = prefix.events
    k = len(evs)
    # time to k
    t_k = float(evs[-1].time) if k > 0 else 0.0
    # inter times
    if k > 1:
        times = np.array([e.time for e in evs])
        diffs = np.diff(times)
        mean_diff = float(np.mean(diffs))
        var_diff = float(np.var(diffs))
        # half life
        half_k = k // 2
        t_half = float(times[half_k - 1]) if half_k >= 1 else 0.0
        half_ratio = t_half / t_k if t_k > 0 else 0.0
        # speed change: average inter time in first half vs second half
        first_diffs = diffs[:half_k] if half_k > 0 else np.array([])
        second_diffs = diffs[half_k:] if half_k < len(diffs) else np.array([])
        if len(second_diffs) == 0:
            speed_change = 0.0
        else:
            first_mean = float(np.mean(first_diffs)) if len(first_diffs) > 0 else mean_diff
            second_mean = float(np.mean(second_diffs))
            speed_change = first_mean / second_mean if second_mean > 0 else 0.0
    else:
        mean_diff = 0.0
        var_diff = 0.0
        half_ratio = 0.0
        speed_change = 0.0
    return {
        "time_to_k": t_k,
        "mean_inter_time": mean_diff,
        "var_inter_time": var_diff,
        "half_life_ratio": half_ratio,
        "speed_change": speed_change,
    }


def extract_features(prefix: Cascade) -> Dict[str, float]:
    """Extract a combined set of temporal and structural features for a prefix.

    This function merges the temporal features with structural metrics computed
    on the cascade tree.
    """
    # ensure events sorted
    prefix.sort_events()
    feat = temporal_features(prefix)
    # structural features
    G = build_tree(prefix)
    struct = structural_metrics(G, prefix.root)
    # add number of nodes (including root) as a feature
    struct["num_nodes"] = float(len(G))
    feat.update(struct)
    return feat
