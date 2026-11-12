"""Two-stage classification and label-shift correction (two-stage diagnostic, 2026-10-29).

Stage 1 separates Normal from attack; stage 2, trained on attack rows only,
names the attack. Their outputs combine into ten-class probabilities
P(Normal) = 1 - p_a and P(c) = p_a q_c. The same decision rule applies to a
flat model through p_a = 1 - P(Normal), q = P(attacks) / p_a, so the
threshold effect can be separated from the two-stage model itself:

    predict Normal if p_a < t, else the most likely attack class.

Label-shift correction is the EM procedure of Saerens, Latinne & Decaestecker
(2002): from unlabelled target rows it estimates the target class priors and
rescales the probabilities by pi_target / pi_source. The class index of
Normal is 0 throughout (hsboa.data puts Normal first).
"""

from __future__ import annotations

import numpy as np

NORMAL = 0


def combine(p_attack: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Ten-class probabilities from stage 1 (P(attack), shape (n,)) and stage 2 (attack classes, (n, K-1))."""
    p = np.asarray(p_attack, dtype=float)[:, None]
    return np.hstack([1.0 - p, p * np.asarray(Q, dtype=float)])


def split(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(p_attack, Q) from K-class probabilities with Normal at column 0."""
    P = np.asarray(P, dtype=float)
    p_a = 1.0 - P[:, NORMAL]
    att = P[:, 1:]
    s = att.sum(axis=1, keepdims=True)
    Q = np.divide(att, s, out=np.full_like(att, 1.0 / att.shape[1]), where=s > 0)
    return p_a, Q


def decide(P: np.ndarray, t: float | None = None) -> np.ndarray:
    """Plain argmax (t is None), or Normal if P(attack) < t and otherwise the most likely attack class."""
    P = np.asarray(P, dtype=float)
    if t is None:
        return P.argmax(axis=1)
    p_a, Q = split(P)
    return np.where(p_a < t, NORMAL, 1 + Q.argmax(axis=1))


def em_label_shift(P: np.ndarray, source_prior, max_iter: int = 200, tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Saerens et al. (2002) EM: returns (adjusted probabilities, estimated target prior)."""
    P = np.asarray(P, dtype=float)
    src = np.clip(np.asarray(source_prior, dtype=float), 1e-12, None)
    pi = src.copy()
    W = P
    for _ in range(max_iter):
        W = P * (pi / src)
        W = W / W.sum(axis=1, keepdims=True)
        new = W.mean(axis=0)
        if np.abs(new - pi).max() < tol:
            pi = new
            break
        pi = new
    return W, pi


def shift_sample(y: np.ndarray, normal_share: float, rng) -> np.ndarray:
    """Row indices with the given Normal share: all Normal rows kept, attack classes subsampled in proportion.

    Mimics the official UNSW-NB15 test partition, which has more Normal traffic (45%) than training (32%)
    but a similar mix of attack classes. Every attack class keeps at least one row.
    """
    rng = np.random.default_rng(rng)
    y = np.asarray(y)
    normal = np.flatnonzero(y == NORMAL)
    att = np.flatnonzero(y != NORMAL)
    want = int(round(len(normal) * (1 - normal_share) / normal_share))
    if want >= len(att):
        return np.sort(np.concatenate([normal, att]))
    ratio = want / len(att)
    keep = [normal]
    for c in np.unique(y[att]):
        members = att[y[att] == c]
        k = max(1, int(round(ratio * len(members))))
        keep.append(rng.choice(members, size=min(k, len(members)), replace=False))
    return np.sort(np.concatenate(keep))


def binary_rates(y: np.ndarray, pred: np.ndarray) -> dict:
    """Detection rate (attacks flagged) and false-alarm rate (Normal flagged) of the Normal-vs-attack view."""
    t, p = np.asarray(y) != NORMAL, np.asarray(pred) != NORMAL
    return {"tpr": float(p[t].mean()) if t.any() else 0.0, "fpr": float(p[~t].mean()) if (~t).any() else 0.0}


def best_threshold(P: np.ndarray, y: np.ndarray, k: int, grid=None) -> float:
    """Threshold t on P(attack) that maximises macro-F1 on (P, y); ties go to the smallest t."""
    from hsboa.decision import macro_f1
    grid = np.linspace(0.05, 0.95, 91) if grid is None else np.asarray(grid)
    scores = [macro_f1(y, decide(P, t), k) for t in grid]
    return float(grid[int(np.argmax(scores))])
