"""Config-driven benchmark experiments: run grid, resumable storage, summaries.

A config (``configs/experiments/*.yaml``) declares functions, dimensions,
transforms, methods, seeds and budget. Every (problem, method, seed) is one
run stored as ``<out>/raw/<problem>/<method>/seed_XX.json``, so an
interrupted experiment resumes where it stopped. ``manifest.json`` records
the config and a hash of the package source; resuming with different code
is refused, so a results folder never mixes code versions.

Pairing: the optimiser seed is the run seed, and every optimiser draws its
initial population first, so methods with the same seed start from the same
population. The function instance (shift o, rotation M) is fixed per
(function, D, transform) from ``transform_seed``, as in CEC.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from hsboa import __version__
from hsboa.benchmarks import BASE, UNTRANSFORMED, Benchmark, make_benchmark
from hsboa.optimizers import get_optimizer
from hsboa.stats import a12, friedman, holm, wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ERROR_FLOOR = 1e-8          # CEC convention: errors below this count as 0
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Config and jobs
# ---------------------------------------------------------------------------

def load_experiment(path: str | Path) -> dict:
    """Read an experiment config and resolve ``seeds_from`` and ``harmonic_config``."""
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if "seeds" not in cfg:
        with open(PROJECT_ROOT / cfg["seeds_from"], encoding="utf-8") as fh:
            cfg["seeds"] = yaml.safe_load(fh)["seeds"]
    base = {}
    if cfg.get("harmonic_config"):
        with open(PROJECT_ROOT / cfg["harmonic_config"], encoding="utf-8") as fh:
            base = yaml.safe_load(fh)["harmonic"]
    for m in cfg["methods"]:
        m.setdefault("params", {})
        if m["optimizer"] == "hsboa":
            m["params"]["harmonic"] = {**base, **m["params"].get("harmonic", {})}
    labels = [m["label"] for m in cfg["methods"]]
    if len(set(labels)) != len(labels):
        raise ValueError("method labels must be unique")
    for a, b in cfg.get("comparisons", []):
        if a not in labels or b not in labels:
            raise ValueError(f"comparison {a!r} vs {b!r} names an unknown method")
    unknown = set(cfg["functions"]) - set(BASE)
    if unknown:
        raise ValueError(f"unknown functions {sorted(unknown)}")
    cfg["_config_text"] = path.read_text(encoding="utf-8")
    return cfg


def transform_tag(t: dict) -> str:
    return ("shifted" if t["shift"] else "plain") + ("_rotated" if t["rotate"] else "")


@dataclass(frozen=True)
class Job:
    function: str
    dim: int
    transform: str
    method: str
    seed: int

    @property
    def problem(self) -> str:
        return f"{self.transform}_D{self.dim}_{self.function}"

    def path(self, out_dir: Path) -> Path:
        slug = re.sub(r"[^A-Za-z0-9.]+", "_", self.method).strip("_")
        return Path(out_dir) / "raw" / self.problem / slug / f"seed_{self.seed:02d}.json"


def jobs(cfg: dict) -> list[Job]:
    """The run grid. Untransformed functions appear under the first transform only,
    since every transform gives the same instance and would count it twice."""
    first = transform_tag(cfg["transforms"][0])
    return [Job(fn, D, transform_tag(t), m["label"], s)
            for D in cfg["dims"] for t in cfg["transforms"] for fn in cfg["functions"]
            if fn not in UNTRANSFORMED or transform_tag(t) == first
            for m in cfg["methods"] for s in cfg["seeds"]]


def problem(cfg: dict, job: Job) -> Benchmark:
    """The fixed function instance for (function, D, transform)."""
    t = next(t for t in cfg["transforms"] if transform_tag(t) == job.transform)
    fi = list(BASE).index(job.function)
    ss = np.random.SeedSequence([cfg["transform_seed"], job.dim, fi, int(t["shift"]), int(t["rotate"])])
    return make_benchmark(job.function, job.dim, np.random.default_rng(ss), t["shift"], t["rotate"])


def checkpoints(pop_size: int, budget: int, k: int) -> np.ndarray:
    """Log-spaced evaluation counts from the initial population to the full budget."""
    return np.unique(np.geomspace(pop_size, budget, k).round().astype(int))


def _downsample(x: np.ndarray, k: int) -> list[float]:
    if len(x) == 0:
        return []
    idx = np.unique(np.linspace(0, len(x) - 1, min(k, len(x))).round().astype(int))
    return [float(v) for v in x[idx]]


def run_job(cfg: dict, job: Job) -> dict:
    """Run one job and return its JSON-serialisable record."""
    method = next(m for m in cfg["methods"] if m["label"] == job.method)
    bm = problem(cfg, job)
    lb, ub = bm.bounds(job.dim)
    budget = cfg["budget_per_dim"] * job.dim
    opt = get_optimizer(method["optimizer"], **method["params"])
    t0 = time.perf_counter()
    res = opt(bm, lb, ub, cfg["pop_size"], budget, job.seed)
    wall = time.perf_counter() - t0
    cp = checkpoints(cfg["pop_size"], budget, cfg["checkpoints"])
    rec = {**asdict(job), "problem": job.problem, "optimizer": method["optimizer"],
           "params": method["params"], "budget": budget, "pop_size": cfg["pop_size"],
           "n_evals": res.n_evals, "iterations": res.iterations, "error": res.best_f,
           "wall_s": wall, "checkpoints": cp.tolist(),
           "curve": [float(res.history[c - 1]) for c in cp],
           "diversity": _downsample(res.diversity, cfg["checkpoints"]),
           "spread": _downsample(res.spread, cfg["checkpoints"])}
    if "h_norm" in res.extra:
        h, m = res.extra["h_norm"], res.extra["move_norm"]
        active = h > 0
        ratio = h[active] / np.maximum(m[active], 1e-300)
        rec["h_ratio_median"] = float(np.median(ratio)) if active.any() else 0.0
        rec["h_norm"] = _downsample(h, cfg["checkpoints"])
        rec["move_norm"] = _downsample(m, cfg["checkpoints"])
    return rec


# ---------------------------------------------------------------------------
# Running and storage
# ---------------------------------------------------------------------------

def code_hash(modules: list[str] | None = None) -> str:
    """SHA-256 over the package source files (stands in for a git commit while working locally).

    ``modules`` (e.g. ["models", "train"]) restricts the hash to the modules an
    experiment depends on, so unrelated new code does not block a resume.
    """
    h = hashlib.sha256()
    files = sorted((PROJECT_ROOT / "hsboa").glob("*.py"))
    if modules is not None:
        files = [p for p in files if p.stem in set(modules)]
    for p in files:
        h.update(p.name.encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def _manifest(cfg: dict) -> dict:
    import scipy
    return {"experiment": cfg["experiment"], "hsboa_version": __version__,
            "code_sha256": code_hash(),
            "config_sha256": hashlib.sha256(cfg["_config_text"].encode()).hexdigest(),
            "config": cfg["_config_text"], "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()}


def _run_and_save(cfg: dict, job: Job, out_dir: Path) -> None:
    rec = run_job(cfg, job)
    path = job.path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec), encoding="utf-8")
    tmp.replace(path)                       # atomic: a killed run never leaves a half-written file


def run_experiment(cfg: dict, out_dir: str | Path, n_jobs: int = -1, verbose: int = 5) -> dict:
    """Run every missing job. Refuses to resume a folder made by different code or config."""
    from joblib import Parallel, delayed

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "manifest.json"
    man = _manifest(cfg)
    if man_path.exists():
        old = json.loads(man_path.read_text(encoding="utf-8"))
        for key in ("code_sha256", "config_sha256"):
            if old.get(key) != man[key]:
                raise RuntimeError(f"{out_dir} was produced with a different {key.split('_')[0]}; "
                                   "use a new output folder instead of mixing results")
        man["started"] = old.get("started")
    else:
        man["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
    man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")

    todo = [j for j in jobs(cfg) if not j.path(out_dir).exists()]
    t0 = time.perf_counter()
    Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(_run_and_save)(cfg, j, out_dir) for j in todo)
    man.update(n_runs=len(jobs(cfg)), n_run_this_call=len(todo),
               wall_s_this_call=round(time.perf_counter() - t0, 1),
               finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")
    return man


def load_runs(out_dir: str | Path) -> pd.DataFrame:
    recs = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(Path(out_dir).glob("raw/*/*/seed_*.json"))]
    if not recs:
        raise FileNotFoundError(f"no runs under {out_dir}/raw")
    df = pd.DataFrame(recs)
    df["error_f"] = np.where(df["error"] < ERROR_FLOOR, 0.0, df["error"])
    return df


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per problem and method: median / mean / std / IQR / best / worst of the floored error."""
    g = df.groupby(["problem", "transform", "dim", "function", "method"])["error_f"]
    out = g.agg(n="count", median="median", mean="mean", std="std", best="min", worst="max",
                q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75)).reset_index()
    out["wall_s"] = df.groupby(["problem", "method"])["wall_s"].mean().reindex(
        pd.MultiIndex.from_frame(out[["problem", "method"]])).to_numpy()
    if "h_ratio_median" in df:
        out["h_ratio_median"] = df.groupby(["problem", "method"])["h_ratio_median"].median().reindex(
            pd.MultiIndex.from_frame(out[["problem", "method"]])).to_numpy()
    return out


def paired_tests(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Wilcoxon per (pair, problem), paired by seed; Holm over the whole family.

    ``outcome`` is from the first method's view: '+' significantly better
    (adjusted p < 0.05 and A12 > 0.5), '-' significantly worse, '=' otherwise.
    """
    rows = []
    for problem_id, sub in df.groupby("problem"):
        wide = sub.pivot(index="seed", columns="method", values="error_f")
        for a, b in pairs:
            if a not in wide or b not in wide:
                continue
            x, y = wide[a].dropna(), wide[b].dropna()
            common = x.index.intersection(y.index)
            x, y = x[common].to_numpy(), y[common].to_numpy()
            rows.append({"problem": problem_id, "a": a, "b": b, "n": len(common),
                         "median_a": float(np.median(x)), "median_b": float(np.median(y)),
                         "p": wilcoxon(x, y), "a12": a12(x, y)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = holm(out["p"].to_numpy())
    sig = out["p_holm"] < ALPHA
    out["outcome"] = np.where(sig & (out["a12"] > 0.5), "+", np.where(sig & (out["a12"] < 0.5), "-", "="))
    return out


def win_tie_loss(tests: pd.DataFrame) -> pd.DataFrame:
    if tests.empty:
        return tests
    c = tests.groupby(["a", "b"])["outcome"].value_counts().unstack(fill_value=0)
    for k in "+=-":
        if k not in c:
            c[k] = 0
    return c[["+", "=", "-"]].rename(columns={"+": "win", "=": "tie", "-": "loss"}).reset_index()


def rank_table(summary: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Friedman average ranks of median errors over functions, per (D, transform) and overall."""
    rows = []
    groups = [(f"D{d} {t}", s) for (d, t), s in summary.groupby(["dim", "transform"])]
    groups.append(("all", summary))
    for name, s in groups:
        wide = s.pivot(index="problem", columns="method", values="median")[methods].dropna()
        ranks, p = friedman(wide.to_numpy())
        rows.append({"group": name, "problems": len(wide), "friedman_p": p,
                     **{m: r for m, r in zip(methods, ranks)}})
    return pd.DataFrame(rows)


def curves_table(df: pd.DataFrame, field: str = "curve") -> pd.DataFrame:
    """Median (and quartiles) of a per-run curve, per problem and method, in long format."""
    rows = []
    for (problem_id, method), sub in df.groupby(["problem", "method"]):
        arr = np.array(sub[field].tolist(), dtype=float)
        xs = sub["checkpoints"].iloc[0] if field == "curve" else list(range(arr.shape[1]))
        q1, med, q3 = np.quantile(arr, [0.25, 0.5, 0.75], axis=0)
        rows += [{"problem": problem_id, "method": method, "x": x, "q1": a, "median": b, "q3": c}
                 for x, a, b, c in zip(xs, q1, med, q3)]
    return pd.DataFrame(rows)


def term_profile(df: pd.DataFrame, fractions=(0.02, 0.1, 0.25, 0.5, 0.75, 1.0)) -> pd.DataFrame:
    """Median |added term| / |proposed SBOA escape move| at fractions of the run, per method and D."""
    if "h_norm" not in df:
        return pd.DataFrame()
    rows = []
    for (method, dim), sub in df.dropna(subset=["h_ratio_median"]).groupby(["method", "dim"]):
        H = np.array(sub["h_norm"].tolist(), dtype=float)
        r = H / np.maximum(np.array(sub["move_norm"].tolist(), dtype=float), 1e-300)
        row = {"method": method, "dim": dim}
        row.update({f"{q:.0%}": float(np.median(r[:, round(q * (r.shape[1] - 1))])) for q in fractions})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(cfg: dict, out_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Write summary.csv, tests.csv, wtl.csv, ranks.csv, curves*.csv and report.md into ``out_dir``.

    Only runs in the current job grid are used (runs of since-removed
    duplicate problems in older folders are ignored).
    """
    out_dir = Path(out_dir)
    grid = jobs(cfg)
    keep = {(j.problem, j.method, j.seed) for j in grid}
    df = load_runs(out_dir)
    df = df[[k in keep for k in zip(df["problem"], df["method"], df["seed"])]]
    expected = len(grid)
    methods = [m["label"] for m in cfg["methods"]]
    if "comparisons" in cfg:
        pairs = [tuple(p) for p in cfg["comparisons"]]
    else:
        ref = cfg["reference"]
        pairs = [(ref, m) for m in methods if m != ref]
    tables = {"summary": summary_table(df), "tests": paired_tests(df, pairs)}
    tables["wtl"] = win_tie_loss(tables["tests"])
    tables["ranks"] = rank_table(tables["summary"], methods)
    tables["curves"] = curves_table(df, "curve")
    tables["curves_diversity"] = curves_table(df, "diversity")
    tables["term_profile"] = term_profile(df)
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "report.md").write_text(_report(cfg, tables, len(df), expected), encoding="utf-8")
    return tables


def _fmt(x: float) -> str:
    return "0" if x == 0 else f"{x:.2e}"


def _report(cfg, tables, n_runs, expected) -> str:
    s, wtl, ranks = tables["summary"], tables["wtl"], tables["ranks"]
    methods = [m["label"] for m in cfg["methods"]]
    lines = [f"# {cfg['experiment']}: summary", "",
             f"Runs: {n_runs} of {expected}. Budget {cfg['budget_per_dim']} x D evaluations, "
             f"pop_size {cfg['pop_size']}, {len(cfg['seeds'])} paired seeds. "
             f"Errors below {ERROR_FLOOR:g} count as 0. Tests: two-sided Wilcoxon signed-rank "
             f"paired by seed, Holm-adjusted over all {len(tables['tests'])} tests, alpha {ALPHA}; "
             "outcome from the first method's view, using Vargha-Delaney A12 for direction.", "",
             "## Win / tie / loss", "", _md(wtl), "",
             "## Friedman average ranks (median error, lower is better)", "",
             _md(ranks.round(3)), ""]
    if "h_ratio_median" in s:
        h = s.dropna(subset=["h_ratio_median"]).pivot(index="problem", columns="method",
                                                      values="h_ratio_median")
        lines += ["## Size of the added term: median |term| / |SBOA escape move|", "",
                  "Per-run median over active iterations, then median over runs:", "",
                  _md(h.map(lambda v: f"{v:.2%}").reset_index()), "",
                  "At fractions of the run (median over runs and problems):", "",
                  _md(tables["term_profile"].map(lambda v: f"{v:.3%}" if isinstance(v, float) else v)), ""]
    med = s.pivot(index="problem", columns="method", values="median")[methods]
    lines += ["## Median error per problem", "", _md(med.map(_fmt).reset_index()), ""]
    return "\n".join(lines)


def _md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)
