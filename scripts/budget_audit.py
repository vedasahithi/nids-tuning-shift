"""Exact search-budget accounting from the stored trial logs (no retraining).

For every tuner arm and seed: proposals (rows of trials.csv = every candidate submitted, incl. the initial population),
repeated decoded settings within the run, distinct settings, and rows reused from an earlier interrupted
attempt (disk cache; the result equals a fresh training because fits are seeded). Also the budget-response curve: best-so-far search validation
macro-F1 after the first n DISTINCT trainings, n = 50..max, so arms can be compared at an exactly equal
number of distinct trainings (truncation of the longer arms).

Writes results/budget_audit/{accounting.csv,accounting_summary.csv,equal_budget.csv,equal_budget_summary.csv,report.md}.
Usage (from the repository root):  python scripts/budget_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ARMS = {  # (experiment folder, arm folder, name)
    ("e1b_unsw_gbdt", "hsboa", "hgb_hsboa"), ("e1b_unsw_gbdt", "sboa", "hgb_sboa"),
    ("e1b_unsw_gbdt", "random", "hgb_random"),
    ("e6_unsw_shift", "iw_hsboa", "iw_hsboa"), ("e6_unsw_shift", "iw_sboa", "iw_sboa"),
    ("e6_unsw_shift", "iw_pso", "iw_pso"), ("e6_unsw_shift", "iw_random", "iw_random"),
}
CHECK = (50, 100, 200, 300, 373)


def load(exp: str, arm: str):
    for p in sorted((RES / exp / arm).glob("seed_*/trials.csv")):
        yield int(p.parent.name.split("_")[1]), pd.read_csv(p)


def main() -> None:
    rows, eq = [], []
    for exp, arm, name in sorted(ARMS):
        for seed, t in load(exp, arm):
            fcol = "val_macro_f1_weighted" if arm.startswith("iw_") and "val_macro_f1_weighted" in t else "val_macro_f1"
            skip = {"val_macro_f1", "val_macro_f1_weighted", "fitness", "wall_s", "gradient_steps", "trial", "cached"}
            key_cols = [c for c in t.columns if c not in skip]
            first = ~t.duplicated(subset=key_cols, keep="first")        # first occurrence of a decoded setting
            dist = t[first]
            rows.append({"arm": name, "seed": seed, "proposals": len(t),
                         "repeated_settings": int((~first).sum()), "distinct_settings": len(dist),
                         "reused_from_earlier_attempt": int(t["cached"].sum())})
            best = dist[fcol].cummax().to_numpy()
            rec = {"arm": name, "seed": seed}
            for n in CHECK:
                rec[f"best_after_{n}"] = float(best[min(n, len(best)) - 1]) if len(best) else np.nan
                rec[f"reached_{n}"] = len(best) >= n
            eq.append(rec)
    acc = pd.DataFrame(rows)
    eqd = pd.DataFrame(eq)
    out = RES / "budget_audit"
    out.mkdir(parents=True, exist_ok=True)
    acc.to_csv(out / "accounting.csv", index=False)
    eqd.to_csv(out / "equal_budget.csv", index=False)
    s = acc.groupby("arm")[["proposals", "repeated_settings", "distinct_settings", "reused_from_earlier_attempt"]].agg(["mean", "min", "max"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s.to_csv(out / "accounting_summary.csv")
    cols = [c for c in eqd.columns if c.startswith("best_after_")]
    e = eqd.groupby("arm")[cols].agg(["mean", "std"]).round(4)
    e.columns = [f"{a}_{b}" for a, b in e.columns]
    e.to_csv(out / "equal_budget_summary.csv")
    txt = ("# Search-budget accounting from stored trial logs\n\n" + s.round(1).to_string() +
           "\n\nBest-so-far search-validation macro-F1 after n distinct trainings (mean, sd over seeds):\n\n" +
           e.to_string() + "\n")
    (out / "report.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
