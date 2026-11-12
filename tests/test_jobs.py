import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "tools"))

from _jobs import CLAIM, claim, clear_claims, taken  # noqa: E402


def test_claim_is_exclusive_and_cleared(tmp_path):
    a, b = tmp_path / "hsboa" / "seed_00", tmp_path / "random" / "seed_00"
    assert claim(a, True)                     # first worker gets the run
    assert not claim(a, True)                 # a second worker is refused
    assert taken(a, True) and not taken(b, True)
    assert claim(b, True)
    assert clear_claims(tmp_path) == 2
    assert not (a / CLAIM).exists() and claim(a, True)


def test_without_worker_flag_nothing_is_claimed(tmp_path):
    run = tmp_path / "seed_00"
    assert claim(run, False) and claim(run, False)
    assert not taken(run, False) and not run.exists()


def test_launcher_defaults():
    import run_parallel
    assert run_parallel.default_workers(2) >= 1
    assert run_parallel.default_workers(1) >= run_parallel.default_workers(2)
    env = run_parallel.worker_env(3)
    assert env["TF_NUM_INTRAOP_THREADS"] == env["OMP_NUM_THREADS"] == "3"
    assert set(run_parallel.EXPERIMENTS) == {"pilot", "e1", "e1b", "e2", "e6"}
    assert sorted(run_parallel.PILOT_ARMS) == sorted(
        ("adam_default", "adam_balanced", "swarm_hsboa", "swarm_sboa", "swarm_pso",
         "hpo_hsboa", "hpo_sboa", "hpo_random"))
