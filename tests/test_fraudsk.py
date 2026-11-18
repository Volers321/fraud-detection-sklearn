import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from fraudsk import metrics, models
from fraudsk.data import FEATURES, NUMERIC, build_features, simulate, split_by_day


@pytest.fixture(scope="module")
def small():
    df = simulate(n_cards=500, days=90, seed=1)
    return df, build_features(df)


def test_simulation_is_deterministic():
    a = simulate(n_cards=200, days=40, seed=3, mule_from_day=30)
    b = simulate(n_cards=200, days=40, seed=3, mule_from_day=30)
    pd.testing.assert_frame_equal(a, b)


def test_all_patterns_present_and_rare(small):
    df, _ = small
    assert {"card_testing", "takeover", "stolen_card", "mule_cashout"} <= set(df["pattern"])
    assert 0.002 < df["is_fraud"].mean() < 0.02
    assert df["time"].is_monotonic_increasing


def test_mule_cashout_only_in_test_period(small):
    df, _ = small
    tr, va, te = split_by_day(df)
    mule = df["pattern"] == "mule_cashout"
    assert mule[te].any()
    assert not mule[tr].any() and not mule[va].any()
    assert not (tr & va).any() and not (va & te).any() and (tr | va | te).all()


def test_features_complete_and_finite(small):
    df, X = small
    assert list(X.columns) == FEATURES
    assert len(X) == len(df)
    assert np.isfinite(X[NUMERIC].to_numpy()).all()


def test_features_do_not_look_into_the_future(small):
    """A transaction's features must not change when later transactions are removed."""
    df, X = small
    cut = df["time"].quantile(0.6)
    prefix = df[df["time"] < cut]
    Xp = build_features(prefix)
    pd.testing.assert_frame_equal(X.loc[Xp.index], Xp, check_exact=False, atol=1e-9)


def test_velocity_counts_exclude_current_transaction():
    rows = [(0, t, 10.0, 1, 0, np.nan, np.nan, 0, 7, 0, "legit") for t in (0.0, 60.0, 120.0)]
    df = pd.DataFrame(
        rows,
        columns=["card_id", "time", "amount", "channel", "category", "lat", "lon", "is_foreign",
                 "device", "is_fraud", "pattern"],
    )  # fmt: skip
    df["home_lat"], df["home_lon"], df["day"], df["txn_id"] = 52.0, 21.0, 0, np.arange(3)
    X = build_features(df)
    assert X["cnt_10m"].tolist() == [0, 1, 2]
    assert X["new_device"].tolist() == [1, 0, 0]


def test_average_precision_matches_sklearn():
    rng = np.random.default_rng(0)
    for _ in range(5):
        y = (rng.random(3000) < 0.03).astype(int)
        s = np.round(rng.random(3000) + 0.6 * y, 2)  # rounded: many ties
        assert metrics.average_precision(y, s) == pytest.approx(average_precision_score(y, s))


def test_threshold_matches_budget():
    s = np.random.default_rng(0).random(20000)
    thr = metrics.threshold_for_budget(s, 0.01)
    assert (s >= thr).mean() == pytest.approx(0.01, abs=0.001)


def test_evaluate_on_handmade_case():
    df = pd.DataFrame(
        {
            "is_fraud": [1, 1, 0, 0, 0, 1],
            "amount": [100.0, 300.0, 50.0, 50.0, 50.0, 600.0],
            "pattern": ["takeover", "takeover", "legit", "legit", "legit", "mule_cashout"],
            "card_id": [0, 1, 2, 3, 4, 5],
        }
    )
    score = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.2])
    r = metrics.evaluate(df, score, thr=0.75)
    assert r["recall"] == pytest.approx(2 / 3, abs=1e-3)  # results are rounded to 4 decimals
    assert r["precision"] == 1.0
    assert r["net_savings_pln"] == 400 - 2 * metrics.REVIEW_COST_PLN
    assert r["recall_by_pattern"]["mule_cashout"]["recall"] == 0.0
    assert r["pr_auc_known_patterns"] == 1.0


def test_bootstrap_interval_brackets_point_estimate(small):
    df, X = small
    tr, va, te = split_by_day(df)
    y = df["is_fraud"].to_numpy()
    gb = models.hist_gb(max_iter=60).fit(X[tr], y[tr])
    dte = df[te].reset_index(drop=True)
    s = gb.predict_proba(X[te])[:, 1]
    ci = metrics.bootstrap_ci(dte, s, thr=0.5, n_boot=60)
    assert ci["pr_auc_ci95"][0] <= metrics.average_precision(dte["is_fraud"], s) <= ci["pr_auc_ci95"][1] + 0.05


def test_models_beat_the_base_rate(small):
    df, X = small
    tr, va, te = split_by_day(df)
    y = df["is_fraud"].to_numpy()
    # judge on patterns that exist in training; the new attack is unseen by construction
    test_known = te & (df["pattern"] != "mule_cashout").to_numpy()
    base = y[test_known].mean()
    lr = models.logistic().fit(X[tr], y[tr])
    gb = models.hist_gb(max_iter=80).fit(X[tr], y[tr])
    for m in (lr, gb):
        ap = metrics.average_precision(y[test_known], m.predict_proba(X[test_known])[:, 1])
        assert ap > 5 * base


def test_calibrated_probabilities_are_valid(small):
    df, X = small
    tr, va, _ = split_by_day(df)
    y = df["is_fraud"].to_numpy()
    gb = models.hist_gb(max_iter=60).fit(X[tr], y[tr])
    p = models.calibrate(gb, X[va], y[va]).predict_proba(X[va])[:, 1]
    assert p.min() >= 0 and p.max() <= 1


def test_pipeline_smoke():
    from fraudsk.pipeline import run_eval

    res = run_eval(n_cards=400, days=90, seed=2, n_iter=2, n_boot=10)
    assert {"isolation_forest", "logistic", "hist_gb", "hybrid_mean"} <= set(res["models"])
    assert set(res["ablation_hist_gb"]) == {f"without_{g}" for g in models.GROUPS}
    assert 0 <= res["models"]["hist_gb"]["recall"] <= 1
