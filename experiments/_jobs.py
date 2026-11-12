"""Job claiming for parallel workers (tools/run_parallel.py).

With ``--worker``, a runner claims each run folder before working on it by
creating ``<run>/.claim`` atomically; a folder another worker has claimed is
skipped. Several workers started on the same experiment therefore share the
runs without overlap, and whichever worker is free takes the next run.
Claims are kept after a run finishes; the launcher deletes all claims of an
experiment before it starts the workers, so a restart picks up runs that a
stopped worker left unfinished (they resume from their checkpoints).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

CLAIM = ".claim"


def claim(run_dir: Path, enabled: bool) -> bool:
    """True if this process may work on ``run_dir`` (always True without ``--worker``)."""
    if not enabled:
        return True
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(run_dir / CLAIM, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"pid": os.getpid(), "at": time.strftime("%Y-%m-%d %H:%M:%S")}))
    return True


def taken(run_dir: Path, enabled: bool) -> bool:
    """True if another worker has claimed ``run_dir`` (always False without ``--worker``)."""
    return enabled and (run_dir / CLAIM).exists()


def clear_claims(out_dir: Path) -> int:
    """Delete every claim under ``out_dir``; returns how many were removed."""
    n = 0
    for p in Path(out_dir).rglob(CLAIM):
        p.unlink(missing_ok=True)
        n += 1
    return n
