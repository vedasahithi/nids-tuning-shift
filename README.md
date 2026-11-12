# The Data Split Matters More Than the Optimiser: Train-Test Shift and the Limits of Metaheuristic Tuning for Intrusion Detection on UNSW-NB15

Code for the paper of the same title (V. Sahiti Yellanki, Basant Sah).

It contains the harmonic secretary bird optimiser (HSBOA) and the compared optimisers, the detectors, the
train/test shift analysis, and the scripts that build every table and figure of the paper from the stored results.

## 1. Setup

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tools/run_tests.py        # unit tests
```

## 2. Data

Download the two official partitions of UNSW-NB15 from
<https://research.unsw.edu.au/projects/unsw-nb15-dataset> and place them in `data/raw/unsw_nb15/`:

```
data/raw/unsw_nb15/UNSW_NB15_training-set.csv
data/raw/unsw_nb15/UNSW_NB15_testing-set.csv
```

The two file names are swapped in the distributed files; the loader identifies each file by its record count,
so use them as downloaded.

## 3. Rebuild the tables and figures from the stored results

The per-seed results behind the paper are in `results/`. No training is needed for this step.

```bash
python scripts/make_paper_tables.py     # paper/tables  (Tables 1 to 4 and S1)
python scripts/make_figures.py          # paper/figures
python tools/check_manifest.py          # every result file listed in manifest.csv exists
```

`MANIFEST.md` maps each table, figure and analysis of the paper to its command, configuration, seeds and
result files.

## 4. Re-run the experiments

Runs are seeded, checkpointed and can be stopped and restarted. Add `--smoke` to any experiment for a quick
installation check.

```bash
python tools/run_parallel.py e1  --workers 16     # QRNN tuning
python tools/run_parallel.py e1b --workers 16     # joint feature and hyper-parameter search (GBDT)
python tools/run_parallel.py e2  --workers 16     # detectors
python tools/run_parallel.py e6  --workers 16     # shift-aware tuning
python experiments/run_shift.py --final-test      # reads the test labels once, after E6 has finished
```

Analyses on top of the runs:

```bash
python scripts/stats_report.py          # paired tests, bootstrap intervals, Holm correction
python scripts/diagnose_shift.py        # domain classifier, label conflicts
python scripts/domain_auc_grouped.py    # the same AUC with duplicate records controlled
python scripts/split_control.py         # decomposition of the random-split gain
python scripts/ablation_tuning.py       # post hoc tuning ablation
python scripts/make_paper_tables.py && python scripts/make_figures.py
```

The benchmark-function study (26 problems) is run with `experiments/run_benchmarks.py` and
`experiments/run_ablation.py`; the functions and settings are in `BENCHMARKS.md`.

Run times on a 16-core desktop: E1b and E6 take 2-4 hours per run, the CNN-BiLSTM about 7 hours per run, the
analysis scripts minutes.

## Layout

```
hsboa/         optimisers, detectors, shift analysis, metrics and statistics
experiments/   experiment entry points
scripts/       analyses, tables and figures
configs/       all settings (one file per experiment)
tools/         parallel launcher, progress monitor, test runner
tests/         unit tests
results/       stored per-seed results and summaries
```

## License

MIT. UNSW-NB15 is distributed by its authors under their own terms and is not included.
