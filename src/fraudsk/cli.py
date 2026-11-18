"""Command line: ``fraudsk eval`` runs the whole study; ``fraudsk simulate`` dumps the data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import simulate


def plot(res: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(res["models"])
    colors = {"isolation_forest": "#9aa0a6", "logistic": "#1f77b4", "hist_gb": "#d95f02", "hybrid_mean": "#2f855a"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=140)
    x = range(len(names))
    ax[0].bar(
        [i - 0.2 for i in x], [res["models"][n]["pr_auc_known_patterns"] for n in names], 0.4,
        label="known patterns", color="#2b6cb0",
    )  # fmt: skip
    ax[0].bar([i + 0.2 for i in x], [res["models"][n]["pr_auc"] for n in names], 0.4, label="all (incl. new attack)", color="#c05621")  # fmt: skip
    ax[0].set_xticks(list(x), names, rotation=15)
    ax[0].set_ylabel("PR-AUC (test)")
    ax[0].legend(frameon=False)
    ax[0].set_title("Ranking quality")
    patterns = ["card_testing", "takeover", "stolen_card", "mule_cashout"]
    w = 0.8 / len(names)
    for j, n in enumerate(names):
        rec = [res["models"][n]["recall_by_pattern"].get(p, {}).get("recall", 0) for p in patterns]
        shift = (j - (len(names) - 1) / 2) * w
        ax[1].bar([i + shift for i in range(len(patterns))], rec, w, label=n, color=colors.get(n))
    ax[1].set_xticks(range(len(patterns)), patterns, rotation=15)
    ax[1].set_ylabel("recall at 0.5% alert budget")
    ax[1].set_title("Per attack pattern (mule_cashout is new in test)")
    ax[1].legend(frameon=False, ncol=2, loc="upper right", fontsize=8)
    for a in ax:
        a.set_ylim(0, 1.3)
        a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="fraudsk")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval", help="run the full temporal evaluation")
    e.add_argument("--n-cards", type=int, default=2500)
    e.add_argument("--days", type=int, default=90)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--n-iter", type=int, default=12)
    e.add_argument("--n-boot", type=int, default=200)
    e.add_argument("--out", default="results/eval.json")
    e.add_argument("--plot", default="docs/eval.png")
    s = sub.add_parser("simulate", help="write the simulated transactions to CSV")
    s.add_argument("--out", default="transactions.csv")
    s.add_argument("--n-cards", type=int, default=2500)
    s.add_argument("--days", type=int, default=90)
    s.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if a.cmd == "simulate":
        simulate(a.n_cards, a.days, a.seed).to_csv(a.out, index=False)
        print(f"wrote {a.out}")
        return
    from .pipeline import run_eval

    res = run_eval(a.n_cards, a.days, a.seed, a.n_iter, a.n_boot)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    Path(a.plot).parent.mkdir(parents=True, exist_ok=True)
    plot(res, a.plot)
    print(json.dumps({k: res[k] for k in ("models", "cost_rule_hist_gb", "drift_days_83_plus")}, indent=2))
    print(f"\nwrote {a.out} and {a.plot} in {res['runtime_s']} s")


if __name__ == "__main__":
    main()
