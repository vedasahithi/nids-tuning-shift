"""Statistical comparison of paired runs.

Runs of different methods with the same seed start from the same initial
population (every optimiser draws it first from ``rng``), so results are
paired by seed and the Wilcoxon signed-rank test applies.

* :func:`wilcoxon` paired two-sided test; identical samples give p = 1.
* :func:`holm` Holm-Bonferroni step-down adjustment over a family of tests.
* :func:`a12` Vargha-Delaney effect size for minimisation: P(a < b) + 0.5 P(a = b).
  0.5 = no effect; 0.56 / 0.64 / 0.71 are the usual small / medium / large thresholds.
* :func:`friedman` average ranks across problems and the Friedman test.
* :func:`bootstrap_ci` percentile bootstrap CI of the mean paired difference.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as st


def wilcoxon(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired samples must have the same length")
    if np.all(a == b):
        return 1.0
    return float(st.wilcoxon(a, b, zero_method="pratt", alternative="two-sided").pvalue)


def holm(p) -> np.ndarray:
    """Holm-adjusted p-values (monotone, capped at 1), in the input order."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def a12(a, b) -> float:
    """Probability that a run of ``a`` beats (is lower than) a run of ``b``; ties count half."""
    a, b = np.asarray(a, dtype=float)[:, None], np.asarray(b, dtype=float)[None, :]
    return float((a < b).mean() + 0.5 * (a == b).mean())


def bootstrap_ci(diff, level: float = 0.95, n_boot: int = 10000, rng=0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of paired differences (resampling seeds)."""
    d = np.asarray(diff, dtype=float)
    if len(d) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(rng)
    means = d[rng.integers(len(d), size=(n_boot, len(d)))].mean(axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(means, [a, 1 - a])
    return float(lo), float(hi)


def friedman(scores: np.ndarray) -> tuple[np.ndarray, float]:
    """(average rank per method, Friedman p-value) for a (problems, methods) array, lower = better."""
    scores = np.asarray(scores, dtype=float)
    ranks = np.apply_along_axis(st.rankdata, 1, scores)
    if scores.shape[1] < 3 or scores.shape[0] < 2:
        return ranks.mean(axis=0), float("nan")
    p = st.friedmanchisquare(*scores.T).pvalue
    return ranks.mean(axis=0), float(p)
