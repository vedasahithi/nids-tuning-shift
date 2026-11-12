"""Check that every file listed in manifest.csv exists in this repository.

Usage (from the repository root):  python tools/check_manifest.py
Each entry of the `inputs` column (separated by `;`, globs allowed) must match at least one file; entries
starting with `none` are skipped. Exit status 1 if anything is missing.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    missing, checked = [], 0
    with open(ROOT / "manifest.csv", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            for token in [t.strip() for t in row["inputs"].split(";") if t.strip()]:
                if token.startswith("none"):
                    continue
                checked += 1
                if not any(ROOT.glob(token)):
                    missing.append(f"{row['id']}: {token}")
    for m in missing:
        print("MISSING", m)
    print(f"{checked} entries checked, {len(missing)} missing")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
