# benchmarks: summary

Runs: 5460 of 5460. Budget 10000 x D evaluations, pop_size 30, 30 paired seeds. Errors below 1e-08 count as 0. Tests: two-sided Wilcoxon signed-rank paired by seed, Holm-adjusted over all 156 tests, alpha 0.05; outcome from the first method's view, using Vargha-Delaney A12 for direction.

## Win / tie / loss

| a | b | win | tie | loss |
|---|---|---|---|---|
| HSBOA | CSA | 15 | 11 | 0 |
| HSBOA | DE | 5 | 18 | 3 |
| HSBOA | GWO | 25 | 1 | 0 |
| HSBOA | POA | 26 | 0 | 0 |
| HSBOA | PSO | 8 | 16 | 2 |
| HSBOA | SBOA | 0 | 26 | 0 |

## Friedman average ranks (median error, lower is better)

| group | problems | friedman_p | HSBOA | SBOA | PSO | DE | GWO | POA | CSA |
|---|---|---|---|---|---|---|---|---|---|
| D10 shifted | 7 | 0.0 | 2.5 | 2.5 | 3.286 | 2.429 | 5.714 | 6.429 | 5.143 |
| D10 shifted_rotated | 6 | 0.0 | 2.5 | 2.167 | 3.0 | 3.0 | 5.5 | 7.0 | 4.833 |
| D30 shifted | 7 | 0.0 | 2.286 | 2.286 | 3.429 | 3.071 | 5.857 | 6.857 | 4.214 |
| D30 shifted_rotated | 6 | 0.001 | 2.75 | 2.917 | 2.917 | 3.167 | 5.667 | 7.0 | 3.583 |
| all | 26 | 0.0 | 2.5 | 2.462 | 3.173 | 2.904 | 5.692 | 6.808 | 4.462 |

## Size of the added term: median |term| / |SBOA escape move|

Per-run median over active iterations, then median over runs:

| problem | HSBOA |
|---|---|
| shifted_D10_ackley | 0.00% |
| shifted_D10_griewank | 0.00% |
| shifted_D10_rastrigin | 0.00% |
| shifted_D10_rosenbrock | 0.01% |
| shifted_D10_schwefel_222 | 0.00% |
| shifted_D10_schwefel_226 | 0.00% |
| shifted_D10_sphere | 0.00% |
| shifted_D30_ackley | 0.00% |
| shifted_D30_griewank | 0.00% |
| shifted_D30_rastrigin | 0.00% |
| shifted_D30_rosenbrock | 0.00% |
| shifted_D30_schwefel_222 | 0.00% |
| shifted_D30_schwefel_226 | 0.00% |
| shifted_D30_sphere | 0.00% |
| shifted_rotated_D10_ackley | 0.00% |
| shifted_rotated_D10_griewank | 0.00% |
| shifted_rotated_D10_rastrigin | 0.00% |
| shifted_rotated_D10_rosenbrock | 0.01% |
| shifted_rotated_D10_schwefel_222 | 0.00% |
| shifted_rotated_D10_sphere | 0.00% |
| shifted_rotated_D30_ackley | 0.00% |
| shifted_rotated_D30_griewank | 0.00% |
| shifted_rotated_D30_rastrigin | 0.00% |
| shifted_rotated_D30_rosenbrock | 0.00% |
| shifted_rotated_D30_schwefel_222 | 0.00% |
| shifted_rotated_D30_sphere | 0.00% |

At fractions of the run (median over runs and problems):

| method | dim | 2% | 10% | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|---|---|
| HSBOA | 10 | 1.605% | 0.009% | 0.000% | 0.000% | 0.000% | 0.000% |
| HSBOA | 30 | 0.847% | 0.010% | 0.000% | 0.000% | 0.000% | 0.000% |

## Median error per problem

| problem | HSBOA | SBOA | PSO | DE | GWO | POA | CSA |
|---|---|---|---|---|---|---|---|
| shifted_D10_ackley | 0 | 0 | 0 | 0 | 1.20e-02 | 6.52e+00 | 1.40e+00 |
| shifted_D10_griewank | 2.46e-02 | 1.97e-02 | 6.89e-02 | 2.46e-02 | 9.23e-01 | 1.72e+00 | 1.78e-01 |
| shifted_D10_rastrigin | 2.98e+00 | 2.98e+00 | 7.96e+00 | 1.99e+00 | 7.71e+00 | 1.69e+01 | 1.29e+01 |
| shifted_D10_rosenbrock | 1.01e+00 | 1.40e+00 | 7.10e-03 | 6.10e+00 | 6.95e+01 | 3.16e+01 | 3.21e+00 |
| shifted_D10_schwefel_222 | 0 | 0 | 0 | 0 | 2.16e-01 | 1.01e+00 | 4.44e-06 |
| shifted_D10_schwefel_226 | 4.74e+02 | 4.74e+02 | 1.12e+03 | 4.51e+02 | 1.23e+03 | 1.10e+03 | 1.24e+03 |
| shifted_D10_sphere | 0 | 0 | 0 | 0 | 5.79e+00 | 1.36e+01 | 0 |
| shifted_D30_ackley | 1.84e+00 | 1.86e+00 | 2.12e+00 | 1.42e+00 | 6.13e+00 | 1.78e+01 | 5.25e+00 |
| shifted_D30_griewank | 1.23e-02 | 9.86e-03 | 2.09e-02 | 4.47e-02 | 7.90e+00 | 1.52e+02 | 1.23e-02 |
| shifted_D30_rastrigin | 6.62e+01 | 6.52e+01 | 1.02e+02 | 2.53e+01 | 7.17e+01 | 1.80e+02 | 1.32e+02 |
| shifted_D30_rosenbrock | 2.14e+01 | 1.79e+01 | 3.99e+00 | 8.67e+02 | 8.28e+04 | 2.68e+07 | 2.76e+01 |
| shifted_D30_schwefel_222 | 0 | 0 | 0 | 0 | 1.11e+01 | 5.01e+01 | 8.73e+00 |
| shifted_D30_schwefel_226 | 2.79e+03 | 2.89e+03 | 4.94e+03 | 2.83e+03 | 6.01e+03 | 5.09e+03 | 4.83e+03 |
| shifted_D30_sphere | 0 | 0 | 0 | 7.79e-03 | 6.13e+02 | 1.99e+04 | 0 |
| shifted_rotated_D10_ackley | 0 | 0 | 0 | 0 | 1.23e+00 | 5.75e+00 | 1.83e+00 |
| shifted_rotated_D10_griewank | 3.20e-02 | 2.83e-02 | 1.18e-01 | 3.20e-02 | 7.04e-01 | 2.68e+00 | 2.21e-01 |
| shifted_rotated_D10_rastrigin | 8.95e+00 | 9.45e+00 | 1.99e+01 | 9.88e+00 | 1.66e+01 | 2.84e+01 | 2.04e+01 |
| shifted_rotated_D10_rosenbrock | 1.65e+00 | 1.53e+00 | 4.72e-01 | 7.46e+00 | 6.56e+02 | 1.12e+03 | 4.37e+00 |
| shifted_rotated_D10_schwefel_222 | 0 | 0 | 0 | 0 | 1.04e+00 | 3.43e+00 | 2.22e-07 |
| shifted_rotated_D10_sphere | 0 | 0 | 0 | 0 | 9.80e-04 | 1.66e-01 | 0 |
| shifted_rotated_D30_ackley | 1.90e+00 | 2.45e+00 | 2.78e+00 | 1.50e+00 | 7.28e+00 | 1.84e+01 | 6.73e+00 |
| shifted_rotated_D30_griewank | 7.40e-03 | 8.63e-03 | 1.72e-02 | 4.54e-02 | 7.63e+00 | 1.62e+02 | 3.70e-03 |
| shifted_rotated_D30_rastrigin | 5.06e+01 | 6.52e+01 | 1.05e+02 | 3.25e+01 | 7.54e+01 | 1.37e+02 | 1.10e+02 |
| shifted_rotated_D30_rosenbrock | 2.55e+01 | 2.49e+01 | 1.60e+01 | 3.06e+01 | 1.17e+05 | 3.31e+06 | 2.74e+01 |
| shifted_rotated_D30_schwefel_222 | 1.43e+01 | 1.11e+01 | 4.47e+00 | 4.99e+00 | 3.76e+01 | 5.62e+01 | 8.34e+00 |
| shifted_rotated_D30_sphere | 0 | 0 | 0 | 1.80e-03 | 8.37e+02 | 1.42e+04 | 0 |
