"""Run one experiment with several worker processes, to use all CPU cores.

Usage (from the repository root):
    python tools/run_parallel.py e1                   # workers chosen from cores and free RAM
    python tools/run_parallel.py e1 --workers 10      # explicit
    python tools/run_parallel.py e2 --models logreg rf hgb --workers 6
    python tools/run_parallel.py pilot                # one process per arm (8 arms)
    python tools/run_parallel.py e1 --smoke --workers 2

Each training run uses about one core (the QRNN is tiny), so one process
leaves a many-core machine mostly idle. This starts N copies of the runner
with ``--worker``; each claims runs one at a time (experiments/_jobs.py), so
the runs are shared without overlap and a free worker always takes the next
one. Every worker is limited to ``--threads`` CPU threads (TensorFlow, OpenMP,
MKL; default 1). Default N: about 80% of the logical cores, capped by free RAM
(0.7 GB per worker). Per-run results do
not depend on the number of workers (each run is seeded and checkpointed).
When all workers have finished, the summary is written once (``--summarize-only``).

Logs: results/logs/<experiment>_wK.log, and <experiment>_parallel.json with
the worker and thread settings (report them with the timings).
Stop with Ctrl+C (or by closing the window); start the same command again to
continue: claims are cleared and every unfinished run resumes from its checkpoint.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from _jobs import clear_claims  # noqa: E402

PY = sys.executable
# name -> (runner script, results folder, can summarise separately)
EXPERIMENTS = {
    "pilot": ("scripts/pilot_qrnn.py", "pilot_qrnn_unsw", False),
    "e1": ("experiments/run_hpo.py", "e1_unsw_tuners", True),
    "e1b": ("experiments/run_gbdt.py", "e1b_unsw_gbdt", True),
    "e2": ("experiments/run_baselines.py", "e2_unsw_baselines", True),
    "e6": ("experiments/run_shift.py", "e6_unsw_shift", True),
}
PILOT_ARMS = ("swarm_hsboa", "swarm_sboa", "swarm_pso", "hpo_hsboa", "hpo_sboa", "hpo_random",
              "adam_default", "adam_balanced")    # costliest first
# Measured on the i7-13700 PC (2026-10-26): a worker uses ~0.5 GB and less than one core (TensorFlow's
# per-step overhead dominates for these tiny models), so many 1-thread workers beat few 2-thread ones.
GB_PER_WORKER = 0.7


def free_ram_gb() -> float | None:
    if os.name != "nt":
        try:
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
        except (ValueError, OSError, AttributeError):
            return None

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 1e9


def default_workers(threads: int) -> int:
    by_cpu = max(1, int(0.8 * (os.cpu_count() or 1)) // threads)    # leave ~20% for the system
    ram = free_ram_gb()
    by_ram = by_cpu if ram is None else max(1, int((ram - 2.0) // GB_PER_WORKER))
    return min(by_cpu, by_ram)


def worker_env(threads: int) -> dict:
    env = dict(os.environ)
    env.update(TF_NUM_INTRAOP_THREADS=str(threads), TF_NUM_INTEROP_THREADS="1",
               OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads), PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="3")
    return env


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=sorted(EXPERIMENTS))
    ap.add_argument("--workers", type=int, help="number of worker processes (default: from cores and free RAM)")
    ap.add_argument("--threads", type=int, default=1, help="CPU threads per worker (default 1)")
    ap.add_argument("--models", nargs="+", help="E2 only: subset of models")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    script, folder, summarise = EXPERIMENTS[args.experiment]
    folder += "_smoke" if args.smoke else ""
    workers = args.workers or default_workers(args.threads)
    out = ROOT / "results" / folder
    logs = ROOT / "results" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    tag = folder

    base = [PY, "-u", str(ROOT / script)] + (["--smoke"] if args.smoke else [])
    if args.experiment == "pilot":
        workers = min(workers, len(PILOT_ARMS))
        groups = [list(PILOT_ARMS[k::workers]) for k in range(workers)]
        commands = [base + ["--arms"] + g for g in groups]
    else:
        n = clear_claims(out) if out.exists() else 0
        if n:
            print(f"cleared {n} old claims (unfinished runs resume from their checkpoints)")
        extra = ["--models"] + args.models if args.models else []
        commands = [base + ["--worker"] + extra for _ in range(workers)]

    info = {"experiment": args.experiment, "folder": folder, "workers": len(commands), "threads": args.threads,
            "logical_cores": os.cpu_count(), "free_ram_gb_at_start": free_ram_gb(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"), "commands": [" ".join(c[2:]) for c in commands]}
    print(f"{args.experiment}: {len(commands)} workers x {args.threads} threads "
          f"({os.cpu_count()} logical cores, {info['free_ram_gb_at_start'] or 0:.1f} GB RAM free); logs in {logs}")
    env = worker_env(args.threads)
    procs = []
    try:
        for k, cmd in enumerate(commands):
            fh = open(logs / f"{tag}_w{k}.log", "a", encoding="utf-8")
            fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd[2:])}\n")
            fh.flush()
            procs.append((subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT), fh))
            if k < len(commands) - 1:
                time.sleep(5)    # the first worker writes the manifest alone
        while True:
            running = sum(p.poll() is None for p, _ in procs)
            print(f"{time.strftime('%H:%M:%S')}  {running}/{len(procs)} workers running "
                  f"(details: tools/progress.py)", flush=True)
            if running == 0:
                break
            time.sleep(60)
    except KeyboardInterrupt:
        print("stopping workers ...")
        for p, _ in procs:
            p.terminate()
        return 1
    finally:
        for _, fh in procs:
            fh.close()

    codes = [p.returncode for p, _ in procs]
    info.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), return_codes=codes)
    (logs / f"{tag}_parallel.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    if any(codes):
        print(f"some workers failed (return codes {codes}); see the logs. Rerun the same command to continue.")
        return 1
    if summarise:
        subprocess.run([PY, str(ROOT / script), "--summarize-only"] + (["--smoke"] if args.smoke else []),
                       cwd=ROOT, env=env, check=False)
    print("all workers finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
