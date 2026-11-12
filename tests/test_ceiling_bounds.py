import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ceiling_bounds as cb  # noqa: E402


def _brute(M):
    g, k = M.shape
    best = -1.0
    for a in itertools.product(range(k), repeat=g):
        best = max(best, cb.assignment_scores(M, np.array(a))[0])
    return best


def test_majority_is_not_macro_f1_optimal_and_bounds_sandwich():
    # class 1 is rare: predicting the minority label in the mixed group helps macro-F1
    M = np.array([[10, 4, 0], [0, 0, 6], [30, 0, 0], [3, 3, 0]])
    maj, _ = cb.assignment_scores(M, cb.majority(M))
    ach_a = cb.coordinate_ascent(M, cb.majority(M), np.random.default_rng(0))
    ach, _ = cb.assignment_scores(M, ach_a)
    ub = cb.per_class_upper(M).mean()
    opt = _brute(M)
    assert ach > maj + 1e-9
    assert ach <= opt + 1e-12 <= ub + 1e-9
    assert abs(ach - opt) < 1e-9


def test_random_tables_sandwich():
    rng = np.random.default_rng(1)
    for _ in range(20):
        M = rng.integers(0, 6, (5, 3))
        M[M.sum(1) == 0, 0] = 1
        opt = _brute(M)
        ach, _ = cb.assignment_scores(M, cb.coordinate_ascent(M, cb.majority(M), rng))
        ub = cb.per_class_upper(M).mean()
        assert ach <= opt + 1e-12
        assert opt <= ub + 1e-9
