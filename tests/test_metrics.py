import json

import numpy as np
import pytest
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             f1_score, matthews_corrcoef)

from hsboa.metrics import binary_view, evaluate as _evaluate, predict_labels

# A five-class toy problem keeps the hand computations short; class 0 is Normal.
CLASSES = ("Normal", "A", "B", "C", "D")


def evaluate(y_true, y_pred):
    return _evaluate(y_true, y_pred, classes=CLASSES)


# Hand-made 5x5 confusion matrix (rows = true class 0..4).
CM = np.array([
    [50, 2, 0, 0, 0],
    [3, 40, 1, 0, 0],
    [0, 2, 20, 0, 0],
    [1, 0, 0, 8, 1],
    [0, 0, 0, 1, 2],
])


def labels_from_cm(cm):
    t, p = [], []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            t += [i] * cm[i, j]
            p += [j] * cm[i, j]
    return np.array(t), np.array(p)


@pytest.fixture
def yp():
    return labels_from_cm(CM)


def test_confusion_and_per_class_match_hand_computation(yp):
    r = evaluate(*yp)
    assert np.array_equal(r.confusion, CM)
    pc = r.per_class
    assert pc["support"].tolist() == [52, 44, 22, 10, 3]
    assert pc["recall"].tolist() == pytest.approx([50/52, 40/44, 20/22, 8/10, 2/3])
    assert pc["precision"].tolist() == pytest.approx([50/54, 40/44, 20/21, 8/9, 2/3])
    # Specificity of A: TN = 131 - 44 - 4 (FP from column) = 83, FP = 4.
    assert pc.loc["A", "specificity"] == pytest.approx(83 / 87)
    assert r.summary["accuracy"] == pytest.approx(120 / 131)
    assert r.summary["micro_f1"] == pytest.approx(r.summary["accuracy"])


def test_aggregates_agree_with_sklearn(yp):
    t, p = yp
    s = evaluate(t, p).summary
    assert s["macro_f1"] == pytest.approx(f1_score(t, p, average="macro"))
    assert s["weighted_f1"] == pytest.approx(f1_score(t, p, average="weighted"))
    assert s["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(t, p))
    assert s["mcc"] == pytest.approx(matthews_corrcoef(t, p))
    rep = classification_report(t, p, output_dict=True)
    assert s["macro_precision"] == pytest.approx(rep["macro avg"]["precision"])


def test_binary_view_counts_attack_confusions_as_detected(yp):
    b = binary_view(*yp)
    # Attack rows: A 41, B 22, C 9, D 3 predicted as non-normal.
    assert (b["tp"], b["fn"], b["tn"], b["fp"]) == (75, 4, 50, 2)
    assert b["tpr"] == pytest.approx(75 / 79)
    assert b["tnr"] == pytest.approx(50 / 52)
    assert b["fpr"] == pytest.approx(2 / 52)
    assert b["accuracy"] == pytest.approx(125 / 131)


def test_predict_labels_uses_argmax_not_first_column():
    proba = np.array([[0.1, 0.7, 0.1, 0.05, 0.05],
                      [0.6, 0.1, 0.1, 0.1, 0.1],
                      [0.0, 0.0, 0.0, 0.2, 0.8]])
    assert predict_labels(proba).tolist() == [1, 0, 4]
    with pytest.raises(ValueError):
        predict_labels(np.array([0.2, 0.8]))


def test_absent_class_does_not_crash_or_inflate():
    t = np.array([0, 0, 1, 1, 2, 3])      # no D in y_true
    p = np.array([0, 1, 1, 1, 2, 4])      # one false D prediction
    r = evaluate(t, p)
    rare = r.per_class.loc["D"]
    assert (rare["support"], rare["fp"], rare["recall"], rare["f1"]) == (0, 1, 0, 0)
    assert r.summary["balanced_accuracy"] == pytest.approx(np.mean([0.5, 1, 1, 0]))
    assert r.summary["macro_f1_present"] > r.summary["macro_f1"]


def test_perfect_predictions():
    t = np.arange(5).repeat(4)
    s = evaluate(t, t)
    assert s.summary["accuracy"] == s.summary["macro_f1"] == s.summary["mcc"] == 1.0
    assert s.binary["fpr"] == 0.0


def test_length_mismatch_rejected():
    with pytest.raises(ValueError, match="length mismatch"):
        evaluate([0, 1], [0])


def test_row_and_save(yp, tmp_path):
    r = evaluate(*yp)
    row = r.to_row(model="hsboa", seed=3)
    assert row["model"] == "hsboa" and row["seed"] == 3
    assert {f"f1_{c}" for c in CLASSES} <= row.keys()
    assert row["binary_tpr"] == pytest.approx(75 / 79)
    r.save(tmp_path, prefix="val_")
    saved = json.loads((tmp_path / "val_summary.json").read_text())
    assert saved["summary"]["n"] == 131
    assert (tmp_path / "val_confusion.csv").read_text().startswith("true,Normal")
