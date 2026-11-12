"""Show how far the experiments have got, with an ETA (read-only).

Usage (from code/), in a second window while an experiment runs:
    python tools/progress.py              # all experiments, once
    python tools/progress.py --watch      # refresh every 60 s (Ctrl+C to stop)
    python tools/progress.py --watch 300  # refresh every 5 min
    python tools/progress.py --only pilot e1

Only reads results/ and configs/; it never writes anything, so it cannot
affect a running experiment. Planned runs come from the experiment configs.
ETA = average run time of the finished runs of the same tuner/model (from
result.json) times the runs still to do, divided by the number of runs in
progress (parallel workers, tools/run_parallel.py). Before any run of a group has
finished, the ETA is marked "rough": it is estimated from the trial times in
trials.jsonl, or from other groups of the same experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs" / "experiments"
FINAL_FACTOR = 1.2  # tuning run ~ trials x 1.2: the final 200-epoch retraining on the full training part


def _yaml(name: str) -> dict:
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


def _seeds(n: int) -> list[int]:
    return yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))["seeds"][:n]


def _sd(seed: int) -> str:
    return f"seed_{seed:02d}"


def plan_tuners(cfg: dict) -> dict[str, tuple[list[str], int]]:
    """{tuner: (run dirs, expected distinct trials per run)}. Matched compute: ~min(max_evals) distinct."""
    budgets = cfg["hpo"]["max_evals"]
    expected = min(v for v in budgets.values() if v)          # an untuned arm (E1b default) has 0
    return {t: ([f"{t}/{_sd(s)}" for s in _seeds(cfg["n_seeds"])], expected if budgets.get(t) else 0)
            for t in cfg["tuners"]}


def plan_baselines(cfg: dict) -> dict[str, tuple[list[str], int]]:
    out = {}
    for model, m in cfg["models"].items():
        if m["tune"] == "grid":
            n = math.prod(len(v) for v in m["grid"].values())
        elif m["tune"] == "random":
            n = m["max_evals"]
        else:
            n = 0
        out[model] = ([f"{model}/{_sd(s)}" for s in _seeds(cfg["n_seeds"])], n)
    return out


def plan_train_size(cfg: dict) -> dict[str, tuple[list[str], int]]:
    return {m["name"]: ([f"{m['name']}/frac_{f:.2f}/{_sd(s)}" for f in cfg["fractions"]
                         for s in _seeds(cfg["n_seeds"])], 0) for m in cfg["models"]}


def plan_feature_order(cfg: dict) -> dict[str, tuple[list[str], int]]:
    orders = ["mutual_info", "original", "reverse_mi"] + [f"random_{k + 1}" for k in range(cfg["n_random_orders"])]
    return {m["name"]: ([f"{m['name']}/{o}/{_sd(s)}" for o in orders for s in _seeds(cfg["n_seeds"])], 0)
            for m in cfg["models"]}


def plan_arms(cfg: dict) -> dict[str, tuple[list[str], int]]:
    """E6: {arm: (run dirs, expected distinct trials per run)}; the default arm has no search."""
    budgets = cfg["hpo"]["max_evals"]
    expected = min(budgets.values())
    return {a: ([f"{a}/{_sd(s)}" for s in _seeds(cfg["n_seeds"])], 0 if spec["tuner"] == "none" else expected)
            for a, spec in cfg["arms"].items()}


EXPERIMENTS = [  # (key, title, results folder, config, planner); None planner = single report
    ("pilot", "Pilot", "pilot_qrnn_unsw", None, "pilot"),
    ("e1", "E1 tuners", "e1_unsw_tuners", "e1_unsw_tuners.yaml", plan_tuners),
    ("e1b", "E1b boosting tuners", "e1b_unsw_gbdt", "e1b_unsw_gbdt.yaml", plan_tuners),
    ("e2", "E2 baselines", "e2_unsw_baselines", "e2_unsw_baselines.yaml", plan_baselines),
    ("e6", "E6 shift-aware HSBOA", "e6_unsw_shift", "e6_unsw_shift.yaml", plan_arms),
]
# Arms of scripts/pilot_qrnn.py, in the order it runs them.
PILOT_ARMS = ("adam_default", "adam_balanced", "swarm_hsboa", "swarm_sboa", "swarm_pso",
              "hpo_hsboa", "hpo_sboa", "hpo_random")


def _trials(run: Path) -> list[dict]:
    return _trials_file(run / "trials.jsonl")


def _trials_file(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # a line being written right now
            pass
    return out


def _wall(run: Path) -> float | None:
    try:
        return float(json.loads((run / "result.json").read_text(encoding="utf-8"))["wall_s"])
    except (OSError, KeyError, ValueError, TypeError):
        return None


def _newest(folder: Path) -> float | None:
    times = [p.stat().st_mtime for p in folder.rglob("*") if p.is_file()]
    return max(times) if times else None


def _dur(s: float) -> str:
    s = max(s, 0)
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    if s < 172800:
        return f"{s / 3600:.1f} h"
    return f"{s / 86400:.1f} days"


def _bar(frac: float, width: int = 24) -> str:
    n = round(max(0.0, min(1.0, frac)) * width)
    return "[" + "#" * n + "-" * (width - n) + "]"


def _pilot_groups(folder: str) -> list[list[str]]:
    """Arm groups of the current parallel pilot workers, from the last header line of each worker log."""
    groups = []
    for log in sorted((RESULTS / "logs").glob(f"{folder}_w*.log")):
        try:
            heads = [ln for ln in log.read_text(encoding="utf-8", errors="replace").splitlines()
                     if ln.startswith("=== ") and "--arms" in ln]
        except OSError:
            continue
        if heads:
            groups.append([a for a in heads[-1].split("--arms", 1)[1].split() if a in PILOT_ARMS])
    return groups


def report_pilot(title: str, out_dir: Path) -> tuple[list[str], float | None]:
    """Pilot (seed 0): one <arm>.json per finished arm; HPO arms also log trials."""
    run = out_dir / "seed_00"
    if not run.exists():
        return [f"{title:<26} not started"], None
    now = time.time()
    done, walls = [], []
    for arm in PILOT_ARMS:
        try:
            walls.append(float(json.loads((run / f"{arm}.json").read_text(encoding="utf-8"))["wall_s"]))
            done.append(arm)
        except (OSError, KeyError, ValueError):
            pass
    n = len(PILOT_ARMS)
    lines = [f"{title:<26} {_bar(len(done) / n)} {len(done)}/{n} arms ({100 * len(done) / n:.0f}%)"]
    todo = [a for a in PILOT_ARMS if a not in done]
    # Parallel workers (tools/run_parallel.py) each run a group of arms in order; the group is in the
    # header line of the worker's log. Without worker logs, arms run one by one in PILOT_ARMS order.
    groups = _pilot_groups(out_dir.name)
    current = [next(a for a in g if a not in done) for g in groups if any(a not in done for a in g)]
    for arm in current or todo[:1]:
        tr = _trials_file(run / f"{arm}_trials.jsonl")
        lines.append(f"      now: {arm}" + (f", trial {len(tr)}" if tr else ""))
    last = _newest(run)
    if last is not None:
        lines.append(f"    last activity {_dur(now - last)} ago ({datetime.fromtimestamp(last):%d %b %H:%M})")
    if not todo:
        lines.append("    finished")
        return lines, 0.0
    if not walls:
        lines.append("    ETA: unknown until the first arm finishes")
        return lines, None
    eta = sum(walls) / len(walls) * len(todo) / max(1, len(current))
    when = datetime.now() + timedelta(seconds=eta)
    lines.append(f"    ETA: ~{_dur(eta)} (rough: arms differ in cost)  ->  {when:%a %d %b, %H:%M}")
    return lines, eta


def report(title: str, folder: str, cfg_name: str | None, planner) -> tuple[list[str], float | None]:
    out_dir = RESULTS / folder
    if planner == "pilot":
        return report_pilot(title, out_dir)
    if planner is None:  # single-report experiment
        if (out_dir / "report.md").exists():
            return [f"{title:<26} done"], 0.0
        return [f"{title:<26} {'started' if out_dir.exists() else 'not started'}"], None
    if not out_dir.exists():
        return [f"{title:<26} not started"], None

    plan = planner(_yaml(cfg_name))
    now = time.time()
    lines, groups = [], {}
    n_total = n_done = 0
    for group, (runs, expected) in plan.items():
        g = {"done": [], "active": [], "pending": 0, "walls": [], "trial_s": [], "expected": expected}
        for rel in runs:
            run = out_dir / rel
            w = _wall(run)
            if w is not None:
                g["done"].append(rel)
                g["walls"].append(w)
            elif run.exists() and any(run.iterdir()):
                tr = _trials(run)
                g["trial_s"] += [t["wall_s"] for t in tr if "wall_s" in t]
                g["active"].append((rel, len(tr), now - run.stat().st_ctime))
            else:
                g["pending"] += 1
        # trial times of finished runs also inform the per-trial estimate
        for rel in g["done"]:
            g["trial_s"] += [t["wall_s"] for t in _trials(out_dir / rel) if "wall_s" in t]
        groups[group] = g
        n_total += len(runs)
        n_done += len(g["done"])

    all_walls = [w for g in groups.values() for w in g["walls"]]
    eta, rough, unknown = 0.0, False, []
    for group, g in groups.items():
        if g["walls"]:
            per_run = sum(g["walls"]) / len(g["walls"])
        elif g["trial_s"] and g["expected"]:
            per_run, rough = sum(g["trial_s"]) / len(g["trial_s"]) * g["expected"] * FINAL_FACTOR, True
        elif all_walls:
            per_run, rough = sum(all_walls) / len(all_walls), True
        else:
            per_run = None
        g["per_run"] = per_run
        todo = g["pending"] + len(g["active"])
        if todo == 0:
            continue
        if per_run is None:
            unknown.append(group)
            continue
        eta += g["pending"] * per_run
        for _, nt, el in g["active"]:
            if g["expected"]:  # by trials done; robust to restarts, unlike the elapsed time
                eta += per_run * (1 - min(nt / g["expected"], 1.0) / FINAL_FACTOR)
            else:
                eta += max(per_run - el, 0.1 * per_run)

    # progress counts active runs by their trial fraction (capped: the final retraining remains)
    partial = sum(min(n / g["expected"], 1.0) / FINAL_FACTOR if g["expected"] else 0.0
                  for g in groups.values() for _, n, _ in g["active"])
    frac = (n_done + partial) / n_total if n_total else 0.0
    n_active = sum(len(g["active"]) for g in groups.values())
    eta /= max(1, n_active)    # parallel workers (tools/run_parallel.py) share the remaining runs
    lines.append(f"{title:<26} {_bar(frac)} {n_done}/{n_total} runs ({100 * frac:.0f}%)"
                 + (f", {n_active} runs active" if n_active > 1 else ""))
    for group, g in groups.items():
        n = len(g["done"]) + len(g["active"]) + g["pending"]
        avg = f", avg {_dur(g['per_run'])}/run" if g["walls"] else ""
        lines.append(f"    {group:<14} {len(g['done']):>3}/{n} done{avg}")
        for rel, nt, el in g["active"]:
            tr = f", trial {nt}/~{g['expected']}" if g["expected"] else ""
            lines.append(f"      now: {rel}{tr}, started {_dur(el)} ago")

    last = _newest(out_dir)
    if last is not None:
        ago = now - last
        msg = f"    last activity {_dur(ago)} ago ({datetime.fromtimestamp(last):%d %b %H:%M})"
        active = [g for g in groups.values() if g["active"]]
        if active and n_done < n_total:
            # expected gap between file writes: one trial, or one run when there is no trial log
            gaps = [(sum(g["trial_s"]) / len(g["trial_s"])) if g["trial_s"] else g["per_run"] for g in active]
            gaps = [x for x in gaps if x]
            if gaps and ago > 3 * max(gaps) + 600:
                msg += "  <-- WARNING: no progress for much longer than usual; is it still running?"
        lines.append(msg)

    if n_done == n_total:
        lines.append("    finished")
        return lines, 0.0
    if unknown:
        lines.append(f"    ETA: unknown until a run of {', '.join(unknown)} finishes"
                     + (f" (others: ~{_dur(eta)})" if eta else ""))
        return lines, None
    when = datetime.now() + timedelta(seconds=eta)
    lines.append(f"    ETA: ~{_dur(eta)}{' (rough)' if rough else ''}  ->  {when:%a %d %b, %H:%M}")
    return lines, eta


def show(only: list[str] | None) -> None:
    print(f"HSBOA experiment progress   {datetime.now():%a %d %b %Y, %H:%M}   ({RESULTS})")
    print()
    for key, title, folder, cfg_name, planner in EXPERIMENTS:
        if only and key not in only:
            continue
        lines, _ = report(title, folder, cfg_name, planner)
        print("\n".join(lines))
    print()
    print("ETAs assume the run keeps the current speed; only runs on this machine are counted.")
    print("Partial runs (E2 with --models) show as not done.")
    print("Dataset: UNSW-NB15 (official partitions).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", nargs="?", const=60, type=int, metavar="SECONDS")
    ap.add_argument("--only", nargs="+", choices=sorted({e[0] for e in EXPERIMENTS}))
    args = ap.parse_args()
    if not args.watch:
        show(args.only)
        return 0
    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            show(args.only)
            print(f"\nrefreshing every {args.watch} s; Ctrl+C to stop")
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
