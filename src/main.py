"""Entry point for training and evaluating cascade prediction models.

This script ties together the parser, feature extractor, label constructor
and model training components.  It reads cascades from a dataset file,
creates k‑prefixes, builds features and labels, trains models and
reports their performance.

Usage::

    python src/main.py --input data/sample_weibo.txt --ks 5 10

"""

from __future__ import annotations

# Add parent directory to sys.path so that relative imports work when executed as a script
import pathlib
import sys as _sys
_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))

import argparse
from typing import List

import numpy as np

# We import modules via the top‑level 'src' package name.  This works both
# when running as a package (python -m src.main) and as a script (python src/main.py)
from src.parse_data import parse_dataset
from src.prefix import generate_prefix_cascades
from src.features import extract_features
from src.labels import construct_labels
from src.models import train_models, evaluate_models, feature_importances
from src.evaluation import label_distribution, print_results

from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description="Predict cascade growth using early retweets.")
    parser.add_argument('--input', type=str, required=True, help='Path to the input dataset file.')
    parser.add_argument('--ks', type=int, nargs='+', default=[5], help='Values of k (prefix lengths) to evaluate.')
    parser.add_argument('--test_size', type=float, default=0.3, help='Fraction of cascades used as the test set.')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for train/test split.')
    args = parser.parse_args()

    # Parse cascades
    cascades = parse_dataset(args.input)
    if not cascades:
        print("No cascades were parsed.  Please check the input file.")
        return
    # Map from cascade ID to final size
    full_sizes = {c.cid: c.n_events() for c in cascades}
    # Evaluate for each k
    for k in args.ks:
        # Generate k‑prefix cascades
        prefixes = generate_prefix_cascades(cascades, k)
        if not prefixes:
            print(f"k={k}: no cascades have at least {k} retweets. Skipping.")
            continue
        # Extract features
        feature_dicts = [extract_features(p) for p in prefixes]
        feature_names = list(feature_dicts[0].keys())
        X = np.array([[fd[name] for name in feature_names] for fd in feature_dicts], dtype=float)
        # Labels
        y = np.array(construct_labels(prefixes, full_sizes, k), dtype=int)
        dist = label_distribution(y)
        # Skip if only one class
        if len(dist) < 2:
            print(f"k={k}: only one class present ({dist}).  Skipping model training.")
            continue
        # Train/test split by cascade (we ensure no cascade appears in both sets)
        # We'll split indices of prefixes by their cascade IDs
        cids = np.array([p.cid for p in prefixes])
        # Unique cascade IDs in prefixes
        unique_cids = np.unique(cids)
        train_cids, test_cids = train_test_split(unique_cids, test_size=args.test_size, random_state=args.random_state)
        train_mask = np.isin(cids, train_cids)
        test_mask = np.isin(cids, test_cids)
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        # Train models
        models = train_models(X_train, y_train)
        # Evaluate
        results = evaluate_models(X_test, y_test, models)
        # Feature importances from RF
        importances = feature_importances(models['rf'], feature_names, top_n=10)
        # Print results
        print_results(k, len(prefixes), dist, results, importances)


if __name__ == '__main__':
    main()
