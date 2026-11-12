import numpy as np
import pandas as pd
import pytest

from hsboa.data import (CLASSES, FEATURE_COLUMNS, Split, class_counts, load_part, load_xy,
                        make_split, to_class_index)
from hsboa.datasets import DEFAULT_DATASET, get_dataset, load_parts


def test_load_returns_features_and_ten_classes(synthetic_file):
    X, y = load_xy(synthetic_file)
    assert list(X.columns) == list(FEATURE_COLUMNS)          # id and the binary label are not features
    assert len(X) == len(y) == 925
    assert set(y) == set(range(len(CLASSES)))
    raw_first = pd.read_csv(synthetic_file).iloc[0]
    assert CLASSES[y[0]] == raw_first["attack_cat"]


def test_record_count_checked(synthetic_file):
    with pytest.raises(ValueError, match="925 records, expected 175341"):
        load_xy(synthetic_file, expected_records=175341)


def test_missing_column_rejected(tmp_path, synthetic_file):
    bad = tmp_path / "bad.csv"
    pd.read_csv(synthetic_file).drop(columns=["sbytes"]).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_xy(bad)


def test_unknown_label_raises():
    with pytest.raises(ValueError, match="unmapped UNSW-NB15 classes"):
        to_class_index(["Normal", "not_a_class"])


def test_labels_are_stripped_and_normal_is_class_zero():
    assert to_class_index(pd.Series([" Normal", "Worms ", "Generic"])).tolist() == [0, 9, 1]


def test_class_counts_include_empty_classes():
    counts = class_counts(np.array([0, 0, 1]))
    assert list(counts.index) == list(CLASSES)
    assert counts.tolist() == [2, 1] + [0] * 8


def test_registry_has_only_unsw():
    ds = get_dataset()
    assert ds.name == DEFAULT_DATASET == "unsw_nb15"
    assert ds.n_classes == 10 and ds.classes[0] == "Normal"
    with pytest.raises(ValueError, match="unknown dataset"):
        get_dataset("nsl_kdd")


def test_split_is_disjoint_stratified_and_deterministic(synthetic_file):
    _, y = load_xy(synthetic_file)
    a = make_split(y, seed=3, val_size=0.2)
    b = make_split(y, seed=3, val_size=0.2)
    c = make_split(y, seed=4, val_size=0.2)
    assert np.array_equal(a.train_idx, b.train_idx)
    assert not np.array_equal(a.val_idx, c.val_idx)
    assert len(np.intersect1d(a.train_idx, a.val_idx)) == 0
    assert len(a.train_idx) + len(a.val_idx) == len(y)
    # Every class, including the rarest, appears in validation in proportion.
    val_share = np.bincount(y[a.val_idx], minlength=10) / np.bincount(y, minlength=10)
    assert np.allclose(val_share, 0.2, atol=0.02)


def test_train_fractions_are_nested_and_keep_validation_fixed(synthetic_file):
    _, y = load_xy(synthetic_file)
    splits = [make_split(y, seed=1, train_fraction=f) for f in (0.6, 0.8, 1.0)]
    for small, big in zip(splits, splits[1:]):
        assert np.isin(small.train_idx, big.train_idx).all()
        assert np.array_equal(small.val_idx, big.val_idx)
    full = np.bincount(y[splits[-1].train_idx], minlength=10)
    sixty = np.bincount(y[splits[0].train_idx], minlength=10)
    assert np.allclose(sixty / full, 0.6, atol=0.05)          # rare classes round to whole rows


def test_split_round_trip(tmp_path, synthetic_file):
    _, y = load_xy(synthetic_file)
    s = make_split(y, seed=7, train_fraction=0.7)
    s.save(tmp_path / "split.npz")
    r = Split.load(tmp_path / "split.npz")
    assert (r.seed, r.train_fraction) == (7, 0.7)
    assert np.array_equal(r.train_idx, s.train_idx)
    assert np.array_equal(r.val_idx, s.val_idx)


@pytest.mark.realdata
def test_real_partitions_match_official_counts(real_train):
    X, y = load_part("train")
    assert len(y) == 175_341
    assert X.iloc[0][["proto", "service", "state", "sbytes"]].tolist() == ["tcp", "-", "FIN", 258]
    assert class_counts(y).to_dict() == {
        "Normal": 56_000, "Generic": 40_000, "Exploits": 33_393, "Fuzzers": 18_184, "DoS": 12_264,
        "Reconnaissance": 10_491, "Analysis": 2_000, "Backdoor": 1_746, "Shellcode": 1_133, "Worms": 130}
    _, _, X_te, y_te = load_parts(get_dataset())
    assert len(y_te) == 82_332
    assert class_counts(y_te).to_dict() == {
        "Normal": 37_000, "Generic": 18_871, "Exploits": 11_132, "Fuzzers": 6_062, "DoS": 4_089,
        "Reconnaissance": 3_496, "Analysis": 677, "Backdoor": 583, "Shellcode": 378, "Worms": 44}
