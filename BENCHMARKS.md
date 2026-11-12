# Benchmark functions and optimiser settings

Used by `experiments/run_benchmarks.py` and `experiments/run_ablation.py` (configuration:
`configs/experiments/benchmarks.yaml`, `ablation.yaml`). Per-problem results for all 26 problems:
`results/benchmarks/per_function.md` and `.csv` (made by `scripts/benchmark_table.py`); raw statistics
`summary.csv`, paired tests `tests.csv`.

## Functions

All functions take x in R^D and have minimum 0. Search box [-b, b]^D.

| Function | f(x) | b | Unshifted optimum x* |
|---|---|---|---|
| Sphere | sum x_i^2 | 100 | 0 |
| Schwefel 2.22 | sum abs(x_i) + prod abs(x_i) | 10 | 0 |
| Rosenbrock | sum 100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2 | 30 | 1 |
| Rastrigin | 10 D + sum x_i^2 - 10 cos(2 pi x_i) | 5.12 | 0 |
| Ackley | -20 exp(-0.2 sqrt(sum x_i^2 / D)) - exp(sum cos(2 pi x_i) / D) + 20 + e | 32 | 0 |
| Griewank | 1 + sum x_i^2 / 4000 - prod cos(x_i / sqrt(i)) | 600 | 0 |
| Schwefel 2.26 | 418.9828872724338 D - sum x_i sin(sqrt(abs(x_i))) | 500 | 420.9687463 |

## Shifts and rotations

The official CEC data files are not used; the classical functions are transformed in the same way
(`hsboa/benchmarks.py`, `make_benchmark`):

- **Shifted:** f(x - o), with the shift vector o drawn uniformly from [-0.8 b, 0.8 b]^D. For Rosenbrock the
  shift is o - 1, which puts the minimum at o.
- **Shifted and rotated:** f(M (x - o) + x*), with M a random orthogonal matrix of determinant +1
  (`scipy.stats.special_ortho_group`).
- **Schwefel 2.26** is never shifted or rotated: its optimum already lies near the edge of the box.
- One fixed instance per (function, dimension, transform) is drawn from `transform_seed: 2026` and shared by all
  methods and seeds, as in CEC.

This gives 26 problems: 7 functions x 2 dimensions (D = 10, 30) shifted, plus 6 functions (all but
Schwefel 2.26) x 2 dimensions shifted and rotated. Problem names are `shifted_D10_ackley`,
`shifted_rotated_D30_rastrigin`, and so on.

## Protocol

- 30 pre-declared seeds (`configs/data.yaml`); the same seed gives every method the same initial population.
- Budget 10,000 x D function evaluations, initial population included; population 30; the final error
  (best value found) is recorded; 50 log-spaced checkpoints per run.
- Paired two-sided Wilcoxon signed-rank test of HSBOA against each other method on each problem, Holm-corrected
  over the whole family of 156 comparisons (6 methods x 26 problems; `tests.csv`). An outcome is `+` or `-` only if the adjusted p < 0.05 and the A12 effect size agrees; `wtl.csv` counts wins, ties and losses.

## Optimiser settings (`hsboa/optimizers.py`)

| Method | Settings |
|---|---|
| HSBOA | SBOA plus the harmonic term of `configs/hsboa.yaml`: L = 10, F = 2, base period P = 2L = 20, kappa0 = 0.1, linear schedule, forecast mode |
| SBOA | Fu et al. (2024): three hunting stages, camouflage and escape, greedy selection, bound clipping; no extra parameters |
| PSO | global-best PSO with constriction weights: inertia 0.7298, c1 = c2 = 1.49618; velocity clipped to 0.2 of the box width, and set to zero at a wall |
| DE | DE/rand/1/bin, F = 0.5, CR = 0.9 |
| GWO | canonical grey wolf optimiser, no greedy step; the three leaders are the best positions seen so far; a decreases linearly from 2 to 0 |
| POA | pelican optimisation: the prey is a random pelican (as in the authors' code), greedy acceptance after each of the two phases, R = 0.2 |
| CSA | crow search with a greedy memory per crow: awareness probability AP = 0.1, flight length fl = 2 |

The ablation (`results/ablation`) replaces the harmonic term by random noise, momentum, or a forecast from
shuffled history of the same size, at kappa0 = 0.1 and kappa0 = 1.
