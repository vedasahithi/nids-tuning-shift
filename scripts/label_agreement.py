"""Cross-partition label agreement for records with identical 42-feature vectors.

Question: when a test record has an identical-feature twin in the training partition, how often
does the twin carry another label, with what coverage and uncertainty, and is that more often than
between twins inside the training partition?

Pairwise disagreement = probability that a random twin has a label different from the record's own.
  cross   test record vs a random training twin
  within  training record vs a random OTHER training twin (groups with >= 2 training records)
Computed overall and per class of the record's own label. Uncertainty: bootstrap over feature
groups (all rows of a group move together), 2000 resamples. Feature equality is exact equality of
the 42 raw values (hash of the raw frame, no scaling or encoding).

Writes results/label_agreement/{summary.csv,per_class.csv,report.md}.
Usage (from the repository root):  python scripts/label_agreement.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def tables(Xtr, ytr, Xte, yte, k):
    key = np.concatenate([pd.util.hash_pandas_object(Xtr.reset_index(drop=True), index=False).to_numpy(),
                          pd.util.hash_pandas_object(Xte.reset_index(drop=True), index=False).to_numpy()])
    codes, _ = pd.factorize(key)
    g = codes.max() + 1
    ntr, nte = len(Xtr), len(Xte)
    A = np.zeros((g, k)); B = np.zeros((g, k))          # train counts, test counts
    np.add.at(A, (codes[:ntr], ytr), 1)
    np.add.at(B, (codes[ntr:], yte), 1)
    return A, B


def stats(A, B, k):
    """Return dict of numerators/denominators per class (cross and within), summed over groups."""
    N = A.sum(1)
    # cross: test rows of class c in groups with N>0: expected disagreeing = (N - A_c)/N
    m = N > 0
    cross_num = (B[m] * ((N[m, None] - A[m]) / N[m, None])).sum(0)
    cross_den = B[m].sum(0)
    # within: train rows of class c in groups with N>=2: (N - A_c)/(N - 1)
    w = N >= 2
    within_num = (A[w] * ((N[w, None] - A[w]) / (N[w, None] - 1))).sum(0)
    within_den = A[w].sum(0)
    cov_all = B.sum(0)
    return cross_num, cross_den, within_num, within_den, cov_all


def main() -> None:
    from hsboa.datasets import get_dataset
    ds = get_dataset()
    k = len(ds.classes)
    Xtr, ytr = ds.load("train")
    Xte, yte = ds.load("test")
    A, B = tables(Xtr, np.asarray(ytr, int), Xte, np.asarray(yte, int), k)
    cn, cd, wn, wd, cov_all = stats(A, B, k)

    def rate(n, d):
        return n / np.maximum(d, 1)

    # per-class table
    pc = pd.DataFrame({"class": ds.classes, "test_rows": cov_all, "test_rows_with_train_twin": cd,
                       "coverage": cd / np.maximum(cov_all, 1),
                       "cross_disagree": rate(cn, cd), "within_train_rows": wd,
                       "within_disagree": rate(wn, wd)})
    tot_cross = cn.sum() / cd.sum()
    tot_within = wn.sum() / wd.sum()
    # standardise within-train to the class mix of the matched test rows
    std_within = float((rate(wn, wd) * cd).sum() / cd.sum())

    # group bootstrap
    rng = np.random.default_rng(0)
    both = (A.sum(1) > 0) & (B.sum(1) > 0)
    gi_all = np.arange(len(A))
    # resample groups that appear in either partition; each group keeps its train and test counts
    boots = []
    for _ in range(2000):
        s = rng.integers(0, len(A), len(A))
        cn_, cd_, wn_, wd_, _ = stats(A[s], B[s], k)
        r_c, r_w = rate(cn_, cd_), rate(wn_, wd_)
        boots.append([cn_.sum() / cd_.sum(), wn_.sum() / wd_.sum(), (r_w * cd_).sum() / cd_.sum(),
                      cn_.sum() / cd_.sum() - (r_w * cd_).sum() / cd_.sum()] + list(r_c - r_w))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    pc["diff_cross_minus_within"] = rate(cn, cd) - rate(wn, wd)
    pc["diff_ci_lo"], pc["diff_ci_hi"] = lo[4:], hi[4:]
    pc.to_csv(ROOT / "results" / "label_agreement" / "per_class.csv", index=False) if (
        ROOT / "results" / "label_agreement").mkdir(parents=True, exist_ok=True) is None else None

    summ = pd.DataFrame([
        {"quantity": "test rows with a training twin (coverage)", "value": cd.sum() / cov_all.sum(),
         "ci_lo": np.nan, "ci_hi": np.nan},
        {"quantity": "test groups with a training twin / test groups",
         "value": both.sum() / (B.sum(1) > 0).sum(), "ci_lo": np.nan, "ci_hi": np.nan},
        {"quantity": "cross-partition pairwise disagreement (test vs train twin)", "value": tot_cross,
         "ci_lo": lo[0], "ci_hi": hi[0]},
        {"quantity": "within-train pairwise disagreement (all matched classes)", "value": tot_within,
         "ci_lo": lo[1], "ci_hi": hi[1]},
        {"quantity": "within-train disagreement standardised to matched test class mix", "value": std_within,
         "ci_lo": lo[2], "ci_hi": hi[2]},
        {"quantity": "excess disagreement: cross minus standardised within-train",
         "value": tot_cross - std_within, "ci_lo": lo[3], "ci_hi": hi[3]},
    ])
    out = ROOT / "results" / "label_agreement"
    summ.to_csv(out / "summary.csv", index=False)
    (out / "report.md").write_text("# Cross-partition label agreement (identical 42-feature records)\n\n"
        + summ.round(4).to_string(index=False) + "\n\nPer class:\n\n" + pc.round(4).to_string(index=False) + "\n",
        encoding="utf-8")
    print(summ.round(4).to_string(index=False)); print(pc.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
