"""Per-function results of the optimiser comparison on the benchmark functions.

Reads results/benchmarks/summary.csv and tests.csv and writes
  results/benchmarks/per_function.csv   one row per problem (26) with mean +- sd of the final error of every
                                        method over the 30 seeds, and HSBOA versus each method (outcome =, +, -
                                        after Holm correction, as in tests.csv)
  results/benchmarks/per_function.md    the same as a readable table

Problem names: <transform>_D<dim>_<function>; the function definitions, bounds, shifts, rotations and the
optimiser settings are in BENCHMARKS.md.
Usage (from the repository root):  python scripts/benchmark_table.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "benchmarks"
METHODS = ["HSBOA", "SBOA", "PSO", "DE", "GWO", "POA", "CSA"]


def fmt(mean: float, sd: float) -> str:
    return f"{mean:.3g} ± {sd:.2g}"


def main() -> None:
    s = pd.read_csv(R / "summary.csv")
    t = pd.read_csv(R / "tests.csv")
    t = t[t.a == "HSBOA"]
    rows = []
    for prob, g in s.groupby("problem", sort=False):
        g = g.set_index("method")
        row = {"problem": prob, "function": g["function"].iloc[0], "dim": int(g["dim"].iloc[0]),
               "transform": g["transform"].iloc[0], "seeds": int(g["n"].iloc[0])}
        for m in METHODS:
            row[m] = fmt(g.loc[m, "mean"], g.loc[m, "std"])
        for m in METHODS[1:]:
            o = t[(t.problem == prob) & (t.b == m)]
            row[f"HSBOA_vs_{m}"] = o["outcome"].iloc[0] if len(o) else ""
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(R / "per_function.csv", index=False)
    cols = ["problem"] + METHODS + [f"HSBOA_vs_{m}" for m in METHODS[1:]]
    lines = ["# Final error (mean ± sd over 30 seeds) on the 26 benchmark problems", "",
             "HSBOA_vs_X: outcome of the paired Wilcoxon test with Holm correction (`=` no significant "
             "difference, `+` HSBOA lower error, `-` HSBOA higher error; see tests.csv for p-values).", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (R / "per_function.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(df)} problems written to {R}")


if __name__ == "__main__":
    main()
