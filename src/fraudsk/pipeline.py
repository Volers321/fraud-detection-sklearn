"""End-to-end evaluation: simulate -> features -> models -> temporal evaluation.

Protocol
--------
* Days 0-59 train, 60-74 validation (threshold + calibration), 75-89 test.
* The ``mule_cashout`` attack appears only from day 76, so the test period contains a new
  fraud pattern. Overall PR-AUC mixes known and new patterns; ``pr_auc_known_patterns`` and
  the per-pattern recall separate them.
* Alert threshold = the validation score that alarms 0.5% of transactions, applied unchanged
  to the test period.
"""

from __future__ import annotations

import time

import numpy as np

from . import metrics, models
from .data import FEATURES, build_features, simulate, split_by_day


def _pct(val: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Percentile rank of ``x`` within the validation scores ``val``."""
    return np.searchsorted(np.sort(val), x) / len(val)


def _top_budget_recall(df, score, budget=metrics.BUDGET):
    """Recall when the top ``budget`` share of the window is reviewed (rank-based)."""
    k = max(int(len(df) * budget), 1)
    top = np.argsort(-score)[:k]
    y = df["is_fraud"].to_numpy()
    return float(y[top].sum() / max(y.sum(), 1))


def run_eval(n_cards: int = 2500, days: int = 90, seed: int = 0, n_iter: int = 12, n_boot: int = 200) -> dict:
    t0 = time.time()
    df = simulate(n_cards, days, seed)
    X = build_features(df)
    tr, va, te = split_by_day(df)
    y = df["is_fraud"].to_numpy()
    dte = df[te].reset_index(drop=True)
    out: dict = {
        "data": {
            "n_cards": n_cards,
            "days": days,
            "seed": seed,
            "n_transactions": int(len(df)),
            "fraud_rate": round(float(y.mean()), 4),
            "fraud_by_pattern": df.loc[y == 1, "pattern"].value_counts().to_dict(),
            "split_days": {"train": "0-59", "validation": "60-74", "test": "75-89"},
            "n_fraud": {"train": int(y[tr].sum()), "validation": int(y[va].sum()), "test": int(y[te].sum())},
        },
        "models": {},
    }

    scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def add(name, model, proba=True):
        if proba:
            fn = lambda d: model.predict_proba(d)[:, 1]  # noqa: E731
        else:
            fn = lambda d: -model.score_samples(d[models.NUMERIC])  # noqa: E731
        scores[name] = (fn(X[va]), fn(X[te]))

    iso = models.isolation_forest(X[tr])
    add("isolation_forest", iso, proba=False)
    lr = models.logistic().fit(X[tr], y[tr])
    add("logistic", lr)
    best = models.tune_hist_gb(X[tr], y[tr], X[va], y[va], n_iter=n_iter, seed=seed)
    gb = models.hist_gb(**best).fit(X[tr], y[tr])
    add("hist_gb", gb)
    out["hist_gb_params"] = {k: (v if v is not None else "none") for k, v in best.items()}

    # hybrid declared a priori: plain mean of the two percentile ranks (each w.r.t. its own
    # validation distribution). A supervised model is blind to attacks absent from training;
    # the unsupervised one is not, at the price of more false alarms on known patterns.
    g, i = scores["hist_gb"], scores["isolation_forest"]
    scores["hybrid_mean"] = tuple(0.5 * (_pct(g[0], g[k]) + _pct(i[0], i[k])) for k in (0, 1))

    for name, (sv, st) in scores.items():
        thr = metrics.threshold_for_budget(sv)
        out["models"][name] = metrics.evaluate(dte, st, thr)
    for name in ("hist_gb", "hybrid_mean"):
        sv, st = scores[name]
        thr = metrics.threshold_for_budget(sv)
        out["models"][name].update(metrics.bootstrap_ci(dte, st, thr, n_boot=n_boot, seed=seed))

    # cost rule on calibrated probabilities: alert if P(fraud) * amount > review cost
    cal = models.calibrate(gb, X[va], y[va])
    p_te = cal.predict_proba(X[te])[:, 1]
    cost = metrics.evaluate(dte, p_te * dte["amount"].to_numpy(), metrics.REVIEW_COST_PLN)
    out["cost_rule_hist_gb"] = {
        k: cost[k] for k in ("alert_rate", "recall", "precision", "net_savings_pln", "fraud_amount_caught_share")
    }
    out["cost_rule_hist_gb"]["recall_by_pattern"] = cost["recall_by_pattern"]

    # ablation: drop one feature group at a time (same hyper-parameters)
    abl = {}
    for g, cols in models.GROUPS.items():
        keep = [c for c in FEATURES if c not in cols]
        m = models.hist_gb(columns=keep, **best).fit(X.loc[tr, keep], y[tr])
        s_va, s_te = m.predict_proba(X.loc[va, keep])[:, 1], m.predict_proba(X.loc[te, keep])[:, 1]
        r = metrics.evaluate(dte, s_te, metrics.threshold_for_budget(s_va))
        abl[f"without_{g}"] = {
            "pr_auc_known_patterns": r["pr_auc_known_patterns"],
            "recall": r["recall"],
            "recall_by_pattern": {k: v["recall"] for k, v in r["recall_by_pattern"].items()},
        }
    out["ablation_hist_gb"] = abl

    # drift: stale model vs one refreshed with labels up to day 77 (5-day label delay),
    # both judged on days 83+ with the same rank-based alert budget
    late = df["day"].to_numpy() >= 83
    d_late = df[late].reset_index(drop=True)
    refresh = df["day"].to_numpy() < 78
    gb2 = models.hist_gb(**best).fit(X[refresh], y[refresh])
    mule = (d_late["pattern"] == "mule_cashout").to_numpy()
    out["drift_days_83_plus"] = {
        "n_mule_rows": int(mule.sum()),
        "stale_model_recall_top_budget": round(_top_budget_recall(d_late, gb.predict_proba(X[late])[:, 1]), 3),
        "refreshed_model_recall_top_budget": round(_top_budget_recall(d_late, gb2.predict_proba(X[late])[:, 1]), 3),
    }

    out["permutation_importance_val"] = models.permutation_top(gb, X[va], y[va])
    out["runtime_s"] = round(time.time() - t0, 1)
    return out
