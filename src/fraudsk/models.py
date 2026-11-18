"""scikit-learn models for card-fraud scoring.

* ``logistic``: scaled numeric features + one-hot categoricals, balanced class weights.
* ``hist_gb``: HistGradientBoostingClassifier with native categorical support; hyper-parameters
  chosen by a randomised search whose only validation fold is the *later* time block
  (``PredefinedSplit``), scored by PR-AUC.
* ``isolation_forest``: unsupervised baseline that never sees a label.
* calibration: isotonic regression fitted on the validation block on top of a frozen model,
  needed for the cost rule "alert if P(fraud) x amount > review cost".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, FEATURES, NUMERIC

# Feature groups used for the ablation study.
GROUPS = {
    "velocity": ["cnt_10m", "cnt_1h", "cnt_24h", "sum_1h_ratio", "log_minutes_since_prev"],
    "geo": ["dist_home_km", "dist_prev_km", "speed_kmh"],
    "device_ip": ["new_device", "is_foreign"],
    "card_history": ["amount_z", "amount_ratio", "new_category", "n_prev"],
}

SEARCH_SPACE = {
    "learning_rate": [0.03, 0.06, 0.1],
    "max_iter": [150, 300, 500],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [20, 50, 100],
    "l2_regularization": [0.0, 1.0, 10.0],
    "class_weight": [None, "balanced"],
}


def logistic() -> Pipeline:
    prep = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    clf = LogisticRegression(class_weight="balanced", C=0.5, max_iter=2000)
    return Pipeline([("prep", prep), ("clf", clf)])


def hist_gb(columns: list[str] | None = None, **params) -> HistGradientBoostingClassifier:
    cols = columns or FEATURES
    cats = [c for c in CATEGORICAL if c in cols]
    return HistGradientBoostingClassifier(
        categorical_features=cats or None, early_stopping=False, random_state=0, **params
    )


def tune_hist_gb(
    X_tr: pd.DataFrame, y_tr, X_va: pd.DataFrame, y_va, n_iter: int = 12, seed: int = 0
) -> dict:
    """Randomised search; the validation block is the later time period."""
    X = pd.concat([X_tr, X_va])
    y = np.r_[np.asarray(y_tr), np.asarray(y_va)]
    fold = np.r_[np.full(len(X_tr), -1), np.zeros(len(X_va), dtype=int)]
    search = RandomizedSearchCV(
        hist_gb(),
        SEARCH_SPACE,
        n_iter=n_iter,
        scoring="average_precision",
        cv=PredefinedSplit(fold),
        refit=False,  # the final model is fitted on the training block only (threshold is set on validation)
        random_state=seed,
        n_jobs=1,
    )
    search.fit(X, y)
    return search.best_params_


def isolation_forest(X_tr: pd.DataFrame) -> IsolationForest:
    return IsolationForest(n_estimators=200, random_state=0, n_jobs=1).fit(X_tr[NUMERIC])


def calibrate(model, X_va: pd.DataFrame, y_va) -> CalibratedClassifierCV:
    """Isotonic calibration on the validation block, base model frozen."""
    cal = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    return cal.fit(X_va, y_va)


def permutation_top(model, X: pd.DataFrame, y, k: int = 8, seed: int = 0) -> list[dict]:
    from sklearn.inspection import permutation_importance

    r = permutation_importance(
        model, X, y, scoring="average_precision", n_repeats=3, random_state=seed, n_jobs=1
    )
    order = np.argsort(-r.importances_mean)[:k]
    return [
        {"feature": X.columns[i], "pr_auc_drop": round(float(r.importances_mean[i]), 4)} for i in order
    ]
