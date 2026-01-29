"""Helper functions for evaluating datasets and printing results."""

from __future__ import annotations

from typing import List, Dict
from collections import Counter


def label_distribution(labels: List[int]) -> Counter:
    """Return a Counter of label occurrences."""
    return Counter(labels)


def print_results(k: int, num_prefixes: int, distribution: Counter, results: Dict[str, Dict[str, float]], importances: List[tuple]) -> None:
    """Pretty‑print evaluation results for a given k.

    Parameters
    ----------
    k: int
        The prefix length.
    num_prefixes: int
        Number of prefix cascades used.
    distribution: collections.Counter
        Counts of labels (0/1).
    results: dict
        Nested dictionary with metrics for each model.
    importances: list of tuples
        Top feature importances from the random forest.
    """
    print(f"=== Results for k={k} ===")
    print(f"Number of prefix cascades: {num_prefixes}")
    print(f"Label distribution: {dict(distribution)}")
    for model_name, metrics in results.items():
        print(f"{model_name} accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}, AUC: {metrics['auc']:.3f}")
    if importances:
        print("Top feature importances from random forest:")
        for name, score in importances:
            print(f"  {name:20s} {score:.4f}")
    print()
