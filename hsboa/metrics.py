"""Multi-class and binary evaluation metrics.

Definitions:

* Class labels come from ``argmax`` over the class probabilities.
* Per class c (one-vs-rest): TP_c, FN_c, FP_c and TN_c are counted from
  the 5x5 confusion matrix M (rows = true class, columns = predicted).
  recall_c = TPR_c = TP_c / (TP_c + FN_c),
  specificity_c = TNR_c = TN_c / (TN_c + FP_c),
  precision_c = TP_c / (TP_c + FP_c), and F1_c is their harmonic mean.
* Macro average: unweighted mean over classes, so Worms counts as much as
  Normal. Weighted average: mean weighted by class support. Micro average:
  computed from summed counts. For single-label multiclass data,
  micro-P = micro-R = micro-F1 = accuracy.
* Balanced accuracy: macro recall over the classes present in y_true.
* MCC: the multiclass Matthews correlation coefficient (Gorodkin's R_K).
* Binary view (normal vs attack): any attack class counts as positive.
  An attack predicted as a different attack class is still a detected
  attack (TP) here. This is the view in which "TPR" (detection rate) and
  "TNR" (normal traffic passed) are usually reported. It is kept separate
  from the multi-class results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, matthews_corrcoef

from hsboa.data import CLASSES

NORMAL_CLASS = CLASSES.index("Normal")


def predict_labels(proba: np.ndarray) -> np.ndarray:
    """Class index of the highest probability per row."""
    proba = np.asarray(proba)
    if proba.ndim != 2:
        raise ValueError(f"expected (n_samples, n_classes) probabilities, got shape {proba.shape}")
    return proba.argmax(axis=1)


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


@dataclass
class Report:
    """Five-class evaluation of one model on one data part."""

    classes: tuple[str, ...]
    confusion: np.ndarray                     # rows = true, columns = predicted
    per_class: pd.DataFrame                   # support, precision, recall, f1, specificity
    summary: dict[str, float]                 # scalar metrics (multi-class and binary)
    binary: dict[str, float] = field(default_factory=dict)

    def confusion_frame(self) -> pd.DataFrame:
        idx = pd.Index(self.classes, name="true")
        cols = pd.Index(self.classes, name="predicted")
        return pd.DataFrame(self.confusion, index=idx, columns=cols)

    def to_row(self, **context) -> dict:
        """One flat dict for a results CSV: context, summary, binary, per-class F1/recall."""
        row = dict(context)
        row.update(self.summary)
        row.update({f"binary_{k}": v for k, v in self.binary.items()})
        for c in self.classes:
            row[f"f1_{c}"] = float(self.per_class.loc[c, "f1"])
            row[f"recall_{c}"] = float(self.per_class.loc[c, "recall"])
        return row

    def save(self, out_dir: str | Path, prefix: str = "") -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.confusion_frame().to_csv(out / f"{prefix}confusion.csv")
        self.per_class.to_csv(out / f"{prefix}per_class.csv")
        (out / f"{prefix}summary.json").write_text(
            json.dumps({"summary": self.summary, "binary": self.binary}, indent=2))
        return out


def binary_view(y_true: np.ndarray, y_pred: np.ndarray,
                normal_class: int = NORMAL_CLASS) -> dict[str, float]:
    """Normal-vs-attack metrics. Positive = attack (any non-normal class)."""
    t = np.asarray(y_true) != normal_class
    p = np.asarray(y_pred) != normal_class
    tp, tn = int(np.sum(t & p)), int(np.sum(~t & ~p))
    fp, fn = int(np.sum(~t & p)), int(np.sum(t & ~p))
    tpr = _safe_div(tp, tp + fn).item()
    precision = _safe_div(tp, tp + fp).item()
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": _safe_div(tp + tn, tp + tn + fp + fn).item(),
        "tpr": tpr,                                   # detection rate
        "tnr": _safe_div(tn, tn + fp).item(),
        "fpr": _safe_div(fp, fp + tn).item(),         # false-alarm rate
        "precision": precision,
        "f1": _safe_div(2 * precision * tpr, precision + tpr).item(),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             classes: tuple[str, ...] = CLASSES) -> Report:
    """Multi-class report plus the binary normal-vs-attack view.

    Classes absent from ``y_true`` (for example Worms in a small validation
    split) keep support 0 and recall 0 in ``per_class``. They are excluded
    from balanced accuracy and from the macro averages over present classes
    (``*_present``), and included in the plain macro averages.
    """
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"length mismatch: {y_true.shape} vs {y_pred.shape}")
    k = len(classes)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(k))
    n = cm.sum()
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    fn, fp = support - tp, predicted - tp
    tn = n - tp - fn - fp

    precision = _safe_div(tp, predicted)
    recall = _safe_div(tp, support)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    specificity = _safe_div(tn, tn + fp)
    present = support > 0

    per_class = pd.DataFrame(
        {"support": support, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
         "precision": precision, "recall": recall, "f1": f1,
         "specificity": specificity},
        index=pd.Index(classes, name="class"),
    )
    w = _safe_div(support, n) if n else np.zeros(k)
    summary = {
        "n": int(n),
        "accuracy": _safe_div(tp.sum(), n).item(),
        "balanced_accuracy": float(recall[present].mean()) if present.any() else 0.0,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "macro_f1_present": float(f1[present].mean()) if present.any() else 0.0,
        "macro_specificity": float(specificity.mean()),
        "weighted_precision": float((w * precision).sum()),
        "weighted_recall": float((w * recall).sum()),
        "weighted_f1": float((w * f1).sum()),
        "micro_f1": _safe_div(tp.sum(), n).item(),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if n else 0.0,
    }
    return Report(classes=tuple(classes), confusion=cm, per_class=per_class,
                  summary=summary, binary=binary_view(y_true, y_pred))
