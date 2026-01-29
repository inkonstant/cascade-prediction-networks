"""Model training and inference for cascade prediction."""

from __future__ import annotations

from typing import Dict, Tuple, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def train_models(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, object]:
    """Train logistic regression and random forest models.

    Logistic regression uses a standard scaler to normalise features.

    Returns
    -------
    dict
        A dictionary mapping model names to fitted model objects.  The
        logistic regression pipeline includes the scaler.
    """
    models: Dict[str, object] = {}
    # Logistic regression with scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_scaled, y_train)
    models['scaler'] = scaler
    models['logreg'] = logreg
    # Random forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    models['rf'] = rf
    return models


def evaluate_models(X_test: np.ndarray, y_test: np.ndarray, models: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    """Evaluate the trained models on a test set.

    Metrics computed: accuracy, F1 score, and AUC (if both classes are present).
    Returns a nested dictionary keyed by model name then metric name.
    """
    results: Dict[str, Dict[str, float]] = {}
    # logistic regression
    scaler = models['scaler']
    logreg = models['logreg']
    X_test_scaled = scaler.transform(X_test)
    preds = logreg.predict(X_test_scaled)
    probs = logreg.predict_proba(X_test_scaled)[:, 1]
    results['logreg'] = _compute_metrics(y_test, preds, probs)
    # random forest
    rf = models['rf']
    preds_rf = rf.predict(X_test)
    probs_rf = rf.predict_proba(X_test)[:, 1]
    results['rf'] = _compute_metrics(y_test, preds_rf, probs_rf)
    return results


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Helper to compute accuracy, F1 and AUC."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
    # AUC requires both classes to be present
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0
    return {'accuracy': float(acc), 'f1': float(f1), 'auc': float(auc)}


def feature_importances(rf: RandomForestClassifier, feature_names: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
    """Extract and return the top `top_n` feature importances from a trained random forest."""
    importances = rf.feature_importances_
    pairs = list(zip(feature_names, importances))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_n]
