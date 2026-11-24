# Reproduction package

Files that let the budgets, the label-conflict counts and the tree and split contrasts be rebuilt and checked
independently. `python scripts/export_package.py --verify` regenerates every file from the data and the code and
compares it with the stored copy (exit status 1 on any difference); it also compares the validation, official,
R, G and P indices with the SHA-256 values in `results/manifest/split_checksums.csv`.

| File | Content |
|---|---|
| `data_checksums.csv` | SHA-256, size, records and columns of the two UNSW-NB15 files as used. The two file names are swapped in the distributed files: `UNSW_NB15_testing-set.csv` holds the 175,341 records of the official training partition and `UNSW_NB15_training-set.csv` the 82,332 of the official test partition. The loader identifies each file by its record count. |
| `record_groups.csv.gz` | Every record of both partitions (`partition`, `row` = position in the file as loaded), the key of its group of identical 42-feature vectors (`group_key`: 64-bit hash of the raw values; equal keys = equal vectors) and a numeric `group_id` shared across the partitions. These groups are the units of the label-conflict counts (Section 5.3) and of the grouped bootstrap (Section 5.6). 153,684 groups; 53,946 contain test records. |
| `indices/seed_XX.npz` | For seeds 0-9. Positions in the training partition: `train_idx` (140,272 records used for fitting), `val_idx` (35,069 validation records), `search_idx` (the class-stratified search sample, about 30,000 records, drawn with `SeedSequence([seed, 2])`). Positions in the merged data (training rows first, then test rows): `official_`, `R_`, `G_` and `P_` `train` / `test` for the split decomposition of Table 4. |
| `preprocessing_per_seed.json` | Per seed: mutual-information scores of the 42 features (their descending order is the feature order used everywhere), scaler minima and maxima, median-imputation values and the class counts of the training, validation and test parts. The preprocessing is fitted on the training part only. |
| `test_metrics_all_arms.csv` | Every stored run of E1, E1b, E2 and E6 (220 runs): seed, final settings, number of distinct trainings, validation and test metrics, per-class F1 and recall and the binary counts. This is the source of Tables 2 and 3. |

Other outputs used by the tables:

* `results/budget_audit/accounting.csv`: per seed and arm, the number of submitted, repeated and distinct decoded settings (Section 4.3, Tables 1 and 2); the search histories are in the `trials.csv` of each run.
* `results/per_class_eval/`: test predictions (`predictions/<arm>/seed_XX.npy`, class index per test record in file order, `y_test.npy` holds the labels), confusion counts (`confusions.csv`), per-class results with seed and grouped intervals (Table 5) and `refit_check.csv`, which compares every refit with the stored test macro-F1. Ten arms are covered: LightGBM default, HGB default, HGB+HSBOA, HGB+SBOA, HGB+RS, LightGBM+RS, XGBoost+RS, random forest, MLP and CNN-BiLSTM. The tree arms and the MLP reproduce the stored scores exactly; the CNN-BiLSTM refit differs by up to 0.016 (we did not establish the cause; training of this network on the CPU is not bit-reproducible here), so its predictions are included but it is not used in Table 5. The logistic regression and QRNN rows of Table 2 have no stored predictions; their metrics are in `test_metrics_all_arms.csv`.
* `results/split_control_by_class/`: confusion counts of the default LightGBM on the official, R, G and P splits for every seed (Table 4 and Section 5.6).
* `results/stats_report/`: the paired contrasts, intervals, Holm-adjusted p-values and equivalence tests behind Table 3.
