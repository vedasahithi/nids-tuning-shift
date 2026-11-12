"""Tests for hsboa.stats and the experiment runner."""

import json

import numpy as np
import pytest
import yaml

from hsboa.experiment import (Job, checkpoints, jobs, load_experiment, load_runs, problem,
                              run_experiment, run_job, summarize)
from hsboa.experiment import code_hash
from hsboa.stats import a12, bootstrap_ci, friedman, holm, wilcoxon
from hsboa.utils import peak_memory_mb


def test_bootstrap_ci_and_peak_memory():
    assert bootstrap_ci(np.ones(10)) == (1.0, 1.0)
    lo, hi = bootstrap_ci(np.arange(10) - 4.5)
    assert lo < 0 < hi
    assert np.isnan(bootstrap_ci([])[0])
    mb = peak_memory_mb()
    assert mb is None or mb > 1


def test_code_hash_can_be_restricted_to_modules():
    full = code_hash()
    part = code_hash(["models", "train"])
    assert part != full and part == code_hash(["train", "models"])
    assert code_hash([]) != part


# --- stats ------------------------------------------------------------------

def test_holm_matches_hand_computation():
    # sorted p: 0.01*4 = 0.04, 0.02*3 = 0.06, 0.03*2 = 0.06 (monotone), 0.5*1 = 0.5
    assert holm([0.03, 0.01, 0.5, 0.02]).tolist() == pytest.approx([0.06, 0.04, 0.5, 0.06])
    assert holm([0.9, 0.8]).max() <= 1.0


def test_a12_and_wilcoxon():
    a, b = np.arange(10.0), np.arange(10.0) + 100
    assert a12(a, b) == 1.0 and a12(b, a) == 0.0 and a12(a, a) == 0.5
    assert wilcoxon(a, a) == 1.0
    assert wilcoxon(a, b) < 0.01
    with pytest.raises(ValueError):
        wilcoxon(a, b[:5])


def test_friedman_ranks():
    scores = np.array([[1, 2, 3], [1, 3, 2], [1, 2, 3], [1, 2, 3]], dtype=float)
    ranks, p = friedman(scores)
    assert ranks.tolist() == pytest.approx([1.0, 2.25, 2.75])
    assert 0 < p < 0.1


# --- experiment runner --------------------------------------------------------

def _config(tmp_path, **over):
    cfg = {"experiment": "t", "seeds": [0, 1, 2], "dims": [3],
           "functions": ["sphere", "rastrigin"],
           "transforms": [{"shift": True, "rotate": False}], "transform_seed": 1,
           "budget_per_dim": 100, "pop_size": 10, "checkpoints": 5,
           "harmonic_config": "configs/hsboa.yaml", "reference": "HSBOA",
           "methods": [{"label": "HSBOA", "optimizer": "hsboa"},
                       {"label": "SBOA", "optimizer": "sboa"},
                       {"label": "DE", "optimizer": "de"}]}
    cfg.update(over)
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return load_experiment(path)


def test_config_merges_harmonic_defaults(tmp_path):
    cfg = _config(tmp_path, methods=[{"label": "H", "optimizer": "hsboa",
                                      "params": {"harmonic": {"kappa0": 2.0}}}])
    h = cfg["methods"][0]["params"]["harmonic"]
    assert h["kappa0"] == 2.0 and h["L"] == 10 and h["F"] == 2
    with pytest.raises(ValueError, match="unknown method"):
        _config(tmp_path, comparisons=[["HSBOA", "nope"]])


def test_problem_instance_is_shared_by_methods_and_seeds(tmp_path):
    cfg = _config(tmp_path)
    a = problem(cfg, Job("sphere", 3, "shifted", "HSBOA", 0))
    b = problem(cfg, Job("sphere", 3, "shifted", "DE", 7))
    c = problem(cfg, Job("rastrigin", 3, "shifted", "DE", 0))
    assert np.array_equal(a.optimum, b.optimum)
    assert not np.array_equal(a.optimum, c.optimum)


def test_same_seed_gives_same_initial_population(tmp_path):
    cfg = _config(tmp_path)
    recs = [run_job(cfg, Job("sphere", 3, "shifted", m, 1)) for m in ("HSBOA", "SBOA", "DE")]
    assert len({r["curve"][0] for r in recs}) == 1          # best of the shared initial population


def test_run_resume_and_summary(tmp_path):
    cfg = _config(tmp_path)
    out = tmp_path / "res"
    man = run_experiment(cfg, out, n_jobs=1, verbose=0)
    assert man["n_run_this_call"] == len(jobs(cfg)) == 18
    assert run_experiment(cfg, out, n_jobs=1, verbose=0)["n_run_this_call"] == 0
    df = load_runs(out)
    assert len(df) == 18 and (df["n_evals"] == 300).all()
    rec = json.loads(next(out.glob("raw/*/HSBOA/seed_00.json")).read_text())
    assert rec["checkpoints"] == checkpoints(10, 300, 5).tolist() and rec["checkpoints"][-1] == 300
    assert "h_ratio_median" in rec
    t = summarize(cfg, out)
    assert set(t["tests"]["b"]) == {"SBOA", "DE"} and len(t["tests"]) == 4
    assert t["wtl"][["win", "tie", "loss"]].to_numpy().sum() == 4
    assert (out / "report.md").exists()


def test_untransformed_function_is_not_counted_twice(tmp_path):
    cfg = _config(tmp_path, functions=["sphere", "schwefel_226"],
                  transforms=[{"shift": True, "rotate": False}, {"shift": True, "rotate": True}])
    probs = {j.problem for j in jobs(cfg)}
    assert probs == {"shifted_D3_sphere", "shifted_rotated_D3_sphere", "shifted_D3_schwefel_226"}


def test_resume_refuses_different_config(tmp_path):
    out = tmp_path / "res"
    run_experiment(_config(tmp_path), out, n_jobs=1, verbose=0)
    with pytest.raises(RuntimeError, match="different config"):
        run_experiment(_config(tmp_path, budget_per_dim=200), out, n_jobs=1, verbose=0)
