# Final evaluation (lgbm)

Wall time 2.8 min; seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]. Settings fixed before the run (script docstring).

## A. Binary Normal-vs-attack LightGBM, official split (test partition)

| settings | accuracy_mean | accuracy_std | precision_mean | precision_std | tpr_mean | tpr_std | f1_mean | f1_std | fpr_mean | fpr_std |
|---|---|---|---|---|---|---|---|---|---|---|
| default | 0.9023 | 0.0010 | 0.8621 | 0.0015 | 0.9792 | 0.0016 | 0.9169 | 0.0008 | 0.1920 | 0.0026 |
| e2_tuned | 0.8922 | 0.0094 | 0.8465 | 0.0152 | 0.9830 | 0.0045 | 0.9095 | 0.0069 | 0.2189 | 0.0259 |

## B. Protocol effect: official split vs random 70/30 split of the merged data (10 classes)

| settings | split | macro_f1_mean | macro_f1_std | accuracy_mean | accuracy_std | binary_accuracy_mean | binary_accuracy_std | binary_fpr_mean | binary_fpr_std |
|---|---|---|---|---|---|---|---|---|---|
| default | official | 0.5345 | 0.0057 | 0.7268 | 0.0020 | 0.8433 | 0.0014 | 0.3374 | 0.0032 |
| default | random_70_30 | 0.6226 | 0.0060 | 0.8140 | 0.0013 | 0.9413 | 0.0009 | 0.1130 | 0.0020 |
| e2_tuned | official | 0.5291 | 0.0083 | 0.7205 | 0.0125 | 0.8355 | 0.0120 | 0.3583 | 0.0285 |
| e2_tuned | random_70_30 | 0.6198 | 0.0055 | 0.8053 | 0.0067 | 0.9351 | 0.0046 | 0.1345 | 0.0228 |

Gain of the random split over the official split (per-seed differences):

| settings | macro_f1_gain_random_vs_official_mean | macro_f1_gain_random_vs_official_min | macro_f1_gain_random_vs_official_max | accuracy_gain_random_vs_official_mean | accuracy_gain_random_vs_official_min | accuracy_gain_random_vs_official_max |
|---|---|---|---|---|---|---|
| default | 0.0882 | 0.0803 | 0.0951 | 0.0872 | 0.0833 | 0.0898 |
| e2_tuned | 0.0907 | 0.0818 | 0.1090 | 0.0848 | 0.0680 | 0.0936 |

## C. Conflict-aware ceiling (group-majority oracle on identical 42-feature records)

| set | rows | unique_feature_vectors | rows_unavoidably_wrong | ceiling_accuracy | ceiling_macro_f1 |
|---|---|---|---|---|---|
| training partition | 175341 | 101040 | 0.0925 | 0.9075 | 0.7890 |
| test partition | 82332 | 53946 | 0.0640 | 0.9360 | 0.7636 |
| both (merged) | 257673 | 153684 | 0.0861 | 0.9139 | 0.7735 |

Per-class ceiling F1 in ceiling.csv. In-sample bound for accuracy; macro-F1 of the majority oracle is near the bound, not exactly it.

## D. E2: Holm over the H1-H4 design family (16 comparisons)

| hypothesis | metric | a | b | mean_diff | ci95_low | ci95_high | p | p_holm_28_old | p_holm_h1_h4 |
|---|---|---|---|---|---|---|---|---|---|
| H1 | val_macro_f1 | hgb_hsboa | cnn_bilstm | 0.0386 | 0.0251 | 0.0522 | 0.0039 | 0.0586 | 0.0312 |
| H1 | test_macro_f1 | hgb_hsboa | cnn_bilstm | 0.0375 | 0.0246 | 0.0497 | 0.0020 | 0.0547 | 0.0312 |
| H4 | val_macro_f1 | hgb_hsboa | hgb_default | -0.0125 | -0.0192 | -0.0062 | 0.0098 | 0.1367 | 0.0586 |
| H4 | test_macro_f1 | hgb_hsboa | hgb_default | -0.0100 | -0.0183 | -0.0016 | 0.1055 | 0.8438 | 0.4219 |
| H2 | val_macro_f1 | hgb_hsboa | hgb_random | 0.0030 | -0.0041 | 0.0106 | 0.6953 | 1.0000 | 1.0000 |
| H2 | test_macro_f1 | hgb_hsboa | hgb_random | 0.0136 | 0.0013 | 0.0253 | 0.0645 | 0.6445 | 0.3223 |
| H3 | val_macro_f1 | hgb_hsboa | hgb_sboa | -0.0019 | -0.0057 | 0.0016 | 0.6602 | 1.0000 | 1.0000 |
| H3 | test_macro_f1 | hgb_hsboa | hgb_sboa | -0.0013 | -0.0044 | 0.0017 | 0.3477 | 1.0000 | 1.0000 |
| H1 | val_macro_f1 | hgb_hsboa | mlp | 0.0496 | 0.0354 | 0.0636 | 0.0020 | 0.0547 | 0.0312 |
| H1 | test_macro_f1 | hgb_hsboa | mlp | 0.0545 | 0.0459 | 0.0648 | 0.0020 | 0.0547 | 0.0312 |
| H1 | val_macro_f1 | hgb_hsboa | qrnn_hsboa | 0.2000 | 0.1812 | 0.2203 | 0.0020 | 0.0547 | 0.0312 |
| H1 | test_macro_f1 | hgb_hsboa | qrnn_hsboa | 0.1663 | 0.1465 | 0.1863 | 0.0020 | 0.0547 | 0.0312 |
| H1 | val_macro_f1 | hgb_hsboa | qrnn_random | 0.1828 | 0.1738 | 0.1917 | 0.0020 | 0.0547 | 0.0312 |
| H1 | test_macro_f1 | hgb_hsboa | qrnn_random | 0.1529 | 0.1400 | 0.1671 | 0.0020 | 0.0547 | 0.0312 |
| H1 | val_macro_f1 | hgb_hsboa | qrnn_random_d20 | 0.1752 | 0.1579 | 0.1960 | 0.0020 | 0.0547 | 0.0312 |
| H1 | test_macro_f1 | hgb_hsboa | qrnn_random_d20 | 0.1416 | 0.1226 | 0.1633 | 0.0020 | 0.0547 | 0.0312 |
