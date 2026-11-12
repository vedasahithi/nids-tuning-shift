"""Paired statistics: effect sizes with intervals, explicit Holm family, equivalence.

Source: per-seed validation and test macro-F1 stored in results/ (seeds 0-9, paired by seed; the
search subsample, validation split and optimiser seed are shared within a seed).

Outputs (results/stats_report/):
  per_seed.csv              macro-F1 of every arm for every seed (val and test)  -> appendix / repo
  contrasts.csv             HSBOA-tuned GBDT (a) vs b: mean difference a-b, 95% percentile bootstrap CI over
                            seeds (10,000 resamples, seed units), two-sided exact Wilcoxon p, Holm p over the
                            16-comparison family H1-H4 (validation and test; p_holm_h1_h4_16) and over the
                            8-comparison family of each metric (p_holm_h1_h4_by_metric_8; test is the primary one)
  contrasts_e6.csv          H5-H10 (shift-aware experiment, test macro-F1, family of 6): pre-specified one-sided
                            p (H9 two-sided), two-sided p, Holm
  equivalence.csv           for H2, H3, H4: 90% bootstrap CI and paired-t TOST p against margins 0.005/0.01/0.02
  report.md
What the intervals mean: seeds change the validation split, the 30,000-row search sample and the optimiser
randomness. Train and test partitions are fixed, so intervals do not include sampling variation of the
benchmark itself. Ten seeds from one dataset are not ten independent replications.
Usage (from the repository root):  python scripts/stats_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402

from hsboa.stats import bootstrap_ci, holm, wilcoxon  # noqa: E402

SOURCES = {  # arm -> (experiment, column, value)
    "hgb_hsboa": ("e1b_unsw_gbdt", "tuner", "hsboa"), "hgb_sboa": ("e1b_unsw_gbdt", "tuner", "sboa"),
    "hgb_random": ("e1b_unsw_gbdt", "tuner", "random"), "hgb_default": ("e1b_unsw_gbdt", "tuner", "default"),
    "qrnn_hsboa": ("e1_unsw_tuners", "tuner", "hsboa"), "qrnn_random": ("e1_unsw_tuners", "tuner", "random"),
    "qrnn_random_d20": ("e2_unsw_baselines", "model", "qrnn_random_d20"),
    "cnn_bilstm": ("e2_unsw_baselines", "model", "cnn_bilstm"), "mlp": ("e2_unsw_baselines", "model", "mlp"),
}
H = {"qrnn_hsboa": "H1", "qrnn_random": "H1", "qrnn_random_d20": "H1", "cnn_bilstm": "H1", "mlp": "H1",
     "hgb_random": "H2", "hgb_sboa": "H3", "hgb_default": "H4"}
MARGINS = (0.005, 0.01, 0.02)
REF = "hgb_hsboa"


def load() -> pd.DataFrame:
    rows = []
    for arm, (exp, col, val) in SOURCES.items():
        d = pd.read_csv(RES / exp / "runs.csv")
        d = d[d[col] == val]
        for r in d.itertuples():
            rows.append({"arm": arm, "seed": int(r.seed), "val_macro_f1": r.val_macro_f1,
                         "test_macro_f1": r.test_macro_f1})
    return pd.DataFrame(rows)


def tost_p(d: np.ndarray, delta: float) -> float:
    """Paired-t TOST: the larger of the two one-sided p-values (null |mean diff| >= delta)."""
    n, m, s = len(d), d.mean(), d.std(ddof=1)
    if s == 0:
        return 0.0 if abs(m) < delta else 1.0
    se = s / np.sqrt(n)
    p_low = 1 - st.t.cdf((m + delta) / se, n - 1)          # H0: mean <= -delta
    p_up = st.t.cdf((m - delta) / se, n - 1)               # H0: mean >= +delta
    return float(max(p_low, p_up))


# ---- H5-H10 (shift-aware experiment): pre-specified one-sided p-values and two-sided p-values with Holm
E6_PAIRS = [("H5", "iw_hsboa", "std_hsboa", "greater"), ("H6", "iw_hsboa", "iw_random", "greater"),
            ("H7", "iw_hsboa", "iw_sboa", "greater"), ("H8", "iw_hsboa", "iw_pso", "greater"),
            ("H9", "iwtrain_hsboa", "iw_hsboa", "two-sided"), ("H10", "iw_hsboa", "default", "greater")]


def e6_contrasts() -> pd.DataFrame:
    r = pd.read_csv(RES / "e6_unsw_shift" / "runs.csv")
    pre = pd.read_csv(RES / "e6_unsw_shift" / "tests.csv")
    pre = pre[pre.metric == "test_macro_f1"].set_index("hypothesis")
    rows = []
    for h, a, b, alt in E6_PAIRS:
        xa = r[r.arm == a].sort_values("seed").test_macro_f1.to_numpy()
        xb = r[r.arm == b].sort_values("seed").test_macro_f1.to_numpy()
        d = xa - xb
        lo, hi = bootstrap_ci(d)
        rows.append({"hypothesis": h, "a": a, "b": b, "prespecified_test": alt, "mean_diff": d.mean(),
                     "ci95_low": lo, "ci95_high": hi, "seeds_a_better": int((d > 0).sum()),
                     "p_prespecified": float(pre.loc[h, "p"]), "p_two_sided": wilcoxon(xa, xb)})
    out = pd.DataFrame(rows)
    out["p_holm_prespecified"] = holm(out["p_prespecified"].to_numpy())
    out["p_holm_two_sided"] = holm(out["p_two_sided"].to_numpy())
    return out


def main() -> None:
    long = load()
    out = RES / "stats_report"
    out.mkdir(parents=True, exist_ok=True)
    wide = long.pivot(index="seed", columns="arm", values=["val_macro_f1", "test_macro_f1"])
    wide.columns = [f"{a}__{b}" for a, b in wide.columns]
    wide.to_csv(out / "per_seed.csv")
    assert wide.notna().all().all() and len(wide) == 10, "all arms need seeds 0-9"

    rows, eq = [], []
    for metric in ("val_macro_f1", "test_macro_f1"):
        a = wide[f"{metric}__{REF}"].to_numpy()
        for b, hyp in H.items():
            d = a - wide[f"{metric}__{b}"].to_numpy()
            lo, hi = bootstrap_ci(d)
            rows.append({"hypothesis": hyp, "metric": metric, "a": REF, "b": b, "mean_diff": d.mean(),
                         "sd_diff": d.std(ddof=1), "ci95_low": lo, "ci95_high": hi, "p": wilcoxon(a, wide[f"{metric}__{b}"]),
                         "seeds_a_better": int((d > 0).sum())})
            if hyp in ("H2", "H3", "H4"):
                l90, h90 = bootstrap_ci(d, level=0.90)
                rec = {"hypothesis": hyp, "metric": metric, "b": b, "mean_diff": d.mean(), "ci90_low": l90,
                       "ci90_high": h90}
                for m in MARGINS:
                    rec[f"equiv_within_{m}"] = bool(l90 > -m and h90 < m)
                    rec[f"tost_p_{m}"] = tost_p(d, m)
                eq.append(rec)
    c = pd.DataFrame(rows)
    c["p_holm_h1_h4_16"] = holm(c["p"].to_numpy())
    # primary family: the 8 test-macro-F1 comparisons (validation contrasts are a separate, secondary family of 8)
    c["p_holm_h1_h4_by_metric_8"] = c.groupby("metric")["p"].transform(lambda s: holm(s.to_numpy()))
    eqd = pd.DataFrame(eq)
    c.to_csv(out / "contrasts.csv", index=False)
    e6 = e6_contrasts()
    e6.to_csv(out / "contrasts_e6.csv", index=False)
    print(e6.round(4).to_string(index=False))
    eqd.to_csv(out / "equivalence.csv", index=False)

    # consistency with the numbers already in the paper
    old = pd.read_csv(RES / "final_eval" / "e2_holm_family.csv")
    m = c.merge(old, on=["metric", "b"], suffixes=("", "_old"))
    assert len(m) == 16
    print("max |p - p_old| =", float((m["p"] - m["p_old"]).abs().max()),
          "; max |Holm - Holm_old| =", float((m["p_holm_h1_h4_16"] - m["p_holm_h1_h4"]).abs().max()),
          "; max |mean diff - old| =", float((m["mean_diff"] - m["mean_diff_old"]).abs().max()))

    txt = ("# Paired statistics (seeds 0-9)\n\nContrasts: a = hgb_hsboa; difference a - b; 95% percentile bootstrap "
           "CI over seeds; two-sided exact Wilcoxon; Holm over the 16 comparisons.\n\n"
           + c.round(4).to_string(index=False) + "\n\nEquivalence (90% bootstrap CI inside +-margin; TOST p):\n\n"
           + eqd.round(4).to_string(index=False) + "\n")
    (out / "report.md").write_text(txt, encoding="utf-8")
    print(c[c.metric == "test_macro_f1"].round(4).to_string(index=False))
    print(eqd.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
