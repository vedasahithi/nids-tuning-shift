"""Tests for the final-evaluation helpers (scripts/final_eval.py)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import final_eval  # noqa: E402


def test_ceiling_counts_unavoidable_errors():
    X = pd.DataFrame({"a": [1, 1, 1, 2, 3, 3], "b": ["x", "x", "x", "y", "z", "z"]})
    y = np.array([0, 0, 1, 1, 2, 2])
    c = final_eval.ceiling(X, y, ("n", "p", "q"))
    assert c["unique_feature_vectors"] == 3
    assert c["rows_unavoidably_wrong"] == pytest.approx(1 / 6)       # the lone 1 in the first group
    assert c["ceiling_accuracy"] == pytest.approx(5 / 6)
    assert c["ceiling_f1_q"] == pytest.approx(1.0)


def test_ceiling_breaks_ties_to_the_lowest_label():
    X = pd.DataFrame({"a": [5, 5]})
    c = final_eval.ceiling(X, np.array([1, 0]), ("n", "p"))
    assert c["ceiling_accuracy"] == pytest.approx(0.5) and c["ceiling_f1_n"] > 0


def test_h1_h4_family_has_eight_comparisons_per_metric():
    assert len(final_eval.H1_H4) == 8
    assert sorted(set(final_eval.H1_H4.values())) == ["H1", "H2", "H3", "H4"]
