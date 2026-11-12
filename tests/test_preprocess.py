import numpy as np
import pandas as pd
import pytest

from hsboa.data import FEATURE_COLUMNS, load_xy
from hsboa.folds import Fold, load_config, prepare_fold, PROJECT_ROOT
from hsboa.preprocess import FeatureSelector, build_pipeline


@pytest.fixture
def xy(synthetic_file):
    return load_xy(synthetic_file)


def test_scaler_is_fitted_on_training_rows_only(xy):
    X, y = xy
    train = X.iloc[:600]
    pipe = build_pipeline(selection="all", order="original").fit(train, y[:600])
    # The fitted range comes from the training rows only.
    num = [c for c in pipe.named_steps["scale"].feature_names_in_ if c == "dur"]
    col = list(pipe.named_steps["scale"].feature_names_in_).index(num[0])
    assert pipe.named_steps["scale"].data_max_[col] == pytest.approx(train["dur"].max())
    # A value outside the training range is not clipped back into [0, 1].
    extreme = train.iloc[[0]].copy()
    extreme["dur"] = train["dur"].max() * 10
    assert pipe.transform(extreme)["dur"].iloc[0] > 1


def test_unseen_category_does_not_crash(xy):
    X, y = xy
    pipe = build_pipeline(selection="all", order="original").fit(X, y)
    row = X.iloc[[0]].copy()
    row["service"] = "never_seen_service"
    out = pipe.transform(row)
    assert np.isfinite(out.to_numpy()).all()


@pytest.mark.parametrize("method", ["svm_rfe", "mutual_info"])
def test_selector_picks_informative_features_first(xy, method):
    X, y = xy
    pipe = build_pipeline(n_features=4, selection=method, order="mutual_info").fit(X, y)
    sel = pipe.named_steps["select"]
    assert len(sel.selected_features_) == 4
    assert set(sel.selected_features_) >= {"sbytes", "ct_srv_src"}
    assert sel.selected_features_[0] == "sbytes"   # highest MI comes first
    mi = [sel.mutual_info_[f] for f in sel.selected_features_]
    assert mi == sorted(mi, reverse=True)
    assert list(pipe.transform(X).columns) == sel.selected_features_


def test_orderings(xy):
    X, y = xy
    names = {}
    for order in ("original", "reverse", "random"):
        pipe = build_pipeline(n_features=6, selection="mutual_info",
                              order=order, random_state=0).fit(X, y)
        names[order] = pipe.named_steps["select"].selected_features_
    pos = [FEATURE_COLUMNS.index(f) for f in names["original"]]
    assert pos == sorted(pos)
    assert names["reverse"] == names["original"][::-1]
    assert sorted(names["random"]) == sorted(names["original"])


def test_invalid_options_rejected(xy):
    X, y = xy
    with pytest.raises(ValueError):
        FeatureSelector(method="nope").fit(X.select_dtypes("number"), y)
    with pytest.raises(ValueError):
        build_pipeline(categorical_encoding="nope")


def test_prepare_fold_end_to_end(xy, tmp_path):
    X, y = xy
    cfg = load_config(PROJECT_ROOT / "configs" / "data.yaml")
    cfg["features"].update(n_features=5, max_samples=None)
    fold = prepare_fold(X, y, seed=0, train_fraction=0.8, cfg=cfg,
                        X_test=X.iloc[:50], y_test=y[:50])
    assert isinstance(fold, Fold)
    assert fold.X_train.shape == (len(fold.y_train), 5)
    assert fold.X_val.shape[1] == fold.X_test.shape[1] == 5
    assert fold.X_train.dtype == np.float32
    assert len(np.intersect1d(fold.split.train_idx, fold.split.val_idx)) == 0
    out = fold.save(tmp_path / "fold")
    for name in ("split.npz", "pipeline.joblib", "features.json"):
        assert (out / name).exists()
    s = fold.summary()
    assert s["selected_features"] == fold.feature_names
    assert sum(s["class_counts"]["val"].values()) == len(fold.y_val)
