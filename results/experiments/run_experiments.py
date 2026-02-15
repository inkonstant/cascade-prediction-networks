"""Utility script to run cascade growth prediction experiments for multiple k values and random seeds.

This script φορτώνει τα δεδομένα, δημιουργεί prefix‑cascades, εξάγει τα χαρακτηριστικά, φτιάχνει ετικέτες,
εκπαιδεύει το λογιστικό μοντέλο και το τυχαίο δάσος για κάθε k και seed και αποθηκεύει
τις μετρικές σε CSV. Επίσης αποθηκεύει την κατανομή ετικετών και τις σημαντικότερες τυχαίου δάσους.
"""

from __future__ import annotations
import argparse
from collections import Counter
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Εισαγωγές από το υπάρχον repository
from src.parse_data import parse_dataset
from src.prefix import generate_prefix_cascades
from src.features import extract_features
from src.labels import construct_labels
from src.models import train_models, evaluate_models, feature_importances


def run_for_k(cascades: List[object], full_sizes: dict, k: int, seeds: List[int], test_size: float):
    """Τρέχει τα πειράματα για μια δεδομένη τιμή k και λίστα seeds."""
    prefixes = generate_prefix_cascades(cascades, k)
    if not prefixes:
        print(f"k={k}: no cascades have at least {k} events. Skipping.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    feature_dicts = [extract_features(p) for p in prefixes]
    feature_names = list(feature_dicts[0].keys())
    X_full = np.array([[fd[name] for name in feature_names] for fd in feature_dicts], dtype=float)
    y_full = np.array(construct_labels(prefixes, full_sizes, k), dtype=int)
    cids = np.array([p.cid for p in prefixes])
    unique_cids = np.unique(cids)

    metrics_rows = []
    for seed in seeds:
        train_cids, test_cids = train_test_split(unique_cids, test_size=test_size, random_state=seed)
        train_mask = np.isin(cids, train_cids)
        test_mask = np.isin(cids, test_cids)
        X_train, X_test = X_full[train_mask], X_full[test_mask]
        y_train, y_test = y_full[train_mask], y_full[test_mask]

        models = train_models(X_train, y_train)
        results = evaluate_models(X_test, y_test, models)
        for model_name, metrics in results.items():
            metrics_rows.append({
                "k": k,
                "seed": seed,
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "auc": metrics["auc"],
            })

    dist = Counter(y_full)
    label_row = {"k": k, "label_0": dist.get(0, 0), "label_1": dist.get(1, 0)}

    # Σημαντικότερα χαρακτηριστικά από RF (εκπαιδεύουμε σε όλο το dataset)
    models_full = train_models(X_full, y_full)
    importances = feature_importances(models_full["rf"], feature_names, top_n=10)
    imp_rows = [{"k": k, "feature": name, "importance": score} for (name, score) in importances]

    return (pd.DataFrame(metrics_rows),
            pd.DataFrame([label_row]),
            pd.DataFrame(imp_rows))


def main():
    parser = argparse.ArgumentParser(description="Run cascade growth prediction experiments over multiple ks and seeds.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input dataset file.")
    parser.add_argument("--ks", type=int, nargs="+", default=[5], help="List of k values to evaluate.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Random seeds.")
    parser.add_argument("--test_size", type=float, default=0.3, help="Test set size.")
    parser.add_argument("--output_prefix", type=str, default="results", help="Prefix for output CSV files.")
    args = parser.parse_args()

    cascades = parse_dataset(args.input)
    if not cascades:
        print("No cascades were parsed.  Please check the input file.")
        return
    full_sizes = {c.cid: c.n_events() for c in cascades}

    all_metrics = []
    all_labels = []
    all_importances = []
    for k in args.ks:
        metrics_df, labels_df, importances_df = run_for_k(cascades, full_sizes, k, args.seeds, args.test_size)
        if not metrics_df.empty:
            all_metrics.append(metrics_df)
            all_labels.append(labels_df)
            all_importances.append(importances_df)

    if all_metrics:
        pd.concat(all_metrics, ignore_index=True).to_csv(f"{args.output_prefix}.csv", index=False)
    if all_labels:
        pd.concat(all_labels, ignore_index=True).to_csv(f"{args.output_prefix}_label_distribution.csv", index=False)
    if all_importances:
        pd.concat(all_importances, ignore_index=True).to_csv(f"{args.output_prefix}_feature_importances.csv", index=False)

    print("Finished experiments. Results saved to CSV files.")


if __name__ == "__main__":
    main()
