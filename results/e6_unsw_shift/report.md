# E6 shift-aware joint selection (lgbm): summary

Final test done (test labels read once).

Finished runs per arm: {'default': 10, 'e2_lgbm (reference)': 10, 'iw_hsboa': 10, 'iw_pso': 10, 'iw_random': 10, 'iw_sboa': 10, 'iwtrain_hsboa': 10, 'std_hsboa': 10} (planned 10 seeds). Budget {'hsboa': 400, 'sboa': 400, 'pso': 400, 'random': 373} evaluations, pop 10, search on 30000 training rows, final fit on the full training part. Importance weights from a domain classifier (training part vs test features, no test labels), capped at the 99th percentile. Hypotheses H5-H10 fixed in the config; Wilcoxon signed-rank paired by seed in the stated direction, Holm over H5-H10 per metric.

## Mean (sd) by arm

| arm | search_fitness_f1_mean | search_fitness_f1_std | val_macro_f1_mean | val_macro_f1_std | val_macro_f1_weighted_mean | val_macro_f1_weighted_std | test_macro_f1_mean | test_macro_f1_std | test_accuracy_mean | test_accuracy_std | test_binary_accuracy_mean | test_binary_accuracy_std | test_f1_Normal_mean | test_f1_Normal_std | test_f1_Fuzzers_mean | test_f1_Fuzzers_std | n_selected_mean | n_selected_std | distinct_trials_mean | distinct_trials_std | wall_min_mean | wall_min_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| default | nan | nan | 0.6477 | 0.0078 | 0.6825 | 0.0105 | 0.5345 | 0.0057 | 0.7268 | 0.0020 | 0.8433 | 0.0014 | 0.7917 | 0.0022 | 0.3764 | 0.0055 | 20.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2581 | 0.0145 |
| e2_lgbm (reference) | nan | nan | 0.6409 | 0.0079 | nan | nan | 0.5291 | 0.0083 | 0.7205 | 0.0125 | 0.8355 | 0.0120 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| iw_hsboa | 0.6646 | 0.0146 | 0.6456 | 0.0097 | 0.6852 | 0.0099 | 0.5193 | 0.0111 | 0.7172 | 0.0170 | 0.8388 | 0.0137 | 0.7832 | 0.0224 | 0.3878 | 0.0159 | 20.8000 | 3.5214 | 374.2000 | 2.3944 | 144.4111 | 75.8180 |
| iw_pso | 0.6614 | 0.0190 | 0.6435 | 0.0182 | 0.6818 | 0.0215 | 0.5218 | 0.0137 | 0.7100 | 0.0140 | 0.8317 | 0.0104 | 0.7718 | 0.0173 | 0.3870 | 0.0153 | 22.9000 | 2.7669 | 400.0000 | 0.0000 | 155.7250 | 60.1114 |
| iw_random | 0.6409 | 0.0135 | 0.6377 | 0.0086 | 0.6737 | 0.0109 | 0.5206 | 0.0147 | 0.7200 | 0.0155 | 0.8405 | 0.0114 | 0.7862 | 0.0196 | 0.3864 | 0.0096 | 21.7000 | 1.9465 | 373.0000 | 0.0000 | 224.4660 | 37.6225 |
| iw_sboa | 0.6661 | 0.0152 | 0.6453 | 0.0103 | 0.6857 | 0.0119 | 0.5203 | 0.0103 | 0.7156 | 0.0154 | 0.8378 | 0.0120 | 0.7816 | 0.0200 | 0.3881 | 0.0154 | 21.0000 | 2.6667 | 372.1000 | 2.7669 | 129.3686 | 52.9166 |
| iwtrain_hsboa | 0.6632 | 0.0161 | 0.6320 | 0.0151 | 0.6776 | 0.0182 | 0.5144 | 0.0115 | 0.7125 | 0.0133 | 0.8394 | 0.0076 | 0.7848 | 0.0140 | 0.3783 | 0.0150 | 20.4000 | 4.7188 | 374.2000 | 2.3944 | 162.1081 | 46.8558 |
| std_hsboa | 0.6285 | 0.0136 | 0.6466 | 0.0104 | 0.6820 | 0.0133 | 0.5223 | 0.0064 | 0.7108 | 0.0103 | 0.8315 | 0.0088 | 0.7714 | 0.0144 | 0.3853 | 0.0119 | 21.0000 | 3.7417 | 374.2000 | 2.3944 | 177.9197 | 86.6478 |

## Hypotheses

| hypothesis | metric | a | b | alternative | n_seeds | mean_a | mean_b | mean_diff | ci95_low | ci95_high | seeds_a_better | p | a12_a_better | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H5 | test_macro_f1 | iw_hsboa | std_hsboa | greater | 10 | 0.5193 | 0.5223 | -0.0030 | -0.0074 | 0.0022 | 4 | 0.8838 | 0.4000 | 1.0000 |
| H5 | val_macro_f1 | iw_hsboa | std_hsboa | greater | 10 | 0.6456 | 0.6466 | -0.0010 | -0.0065 | 0.0031 | 7 | 0.3125 | 0.4800 | 1.0000 |
| H5 | val_macro_f1_weighted | iw_hsboa | std_hsboa | greater | 10 | 0.6852 | 0.6820 | 0.0032 | -0.0051 | 0.0097 | 9 | 0.0420 | 0.6500 | 0.2100 |
| H6 | test_macro_f1 | iw_hsboa | iw_random | greater | 10 | 0.5193 | 0.5206 | -0.0013 | -0.0103 | 0.0081 | 4 | 0.6875 | 0.4300 | 1.0000 |
| H6 | val_macro_f1 | iw_hsboa | iw_random | greater | 10 | 0.6456 | 0.6377 | 0.0079 | 0.0001 | 0.0156 | 5 | 0.1162 | 0.7200 | 0.5811 |
| H6 | val_macro_f1_weighted | iw_hsboa | iw_random | greater | 10 | 0.6852 | 0.6737 | 0.0115 | 0.0040 | 0.0204 | 7 | 0.0137 | 0.7800 | 0.0820 |
| H7 | test_macro_f1 | iw_hsboa | iw_sboa | greater | 10 | 0.5193 | 0.5203 | -0.0010 | -0.0056 | 0.0036 | 4 | 0.7051 | 0.4650 | 1.0000 |
| H7 | val_macro_f1 | iw_hsboa | iw_sboa | greater | 10 | 0.6456 | 0.6453 | 0.0004 | -0.0031 | 0.0040 | 5 | 0.4414 | 0.5250 | 1.0000 |
| H7 | val_macro_f1_weighted | iw_hsboa | iw_sboa | greater | 10 | 0.6852 | 0.6857 | -0.0005 | -0.0035 | 0.0024 | 4 | 0.6348 | 0.5250 | 1.0000 |
| H8 | test_macro_f1 | iw_hsboa | iw_pso | greater | 10 | 0.5193 | 0.5218 | -0.0025 | -0.0108 | 0.0053 | 5 | 0.6152 | 0.4300 | 1.0000 |
| H8 | val_macro_f1 | iw_hsboa | iw_pso | greater | 10 | 0.6456 | 0.6435 | 0.0021 | -0.0060 | 0.0123 | 4 | 0.5000 | 0.4400 | 1.0000 |
| H8 | val_macro_f1_weighted | iw_hsboa | iw_pso | greater | 10 | 0.6852 | 0.6818 | 0.0035 | -0.0056 | 0.0136 | 5 | 0.3125 | 0.4800 | 1.0000 |
| H9 | test_macro_f1 | iwtrain_hsboa | iw_hsboa | two-sided | 10 | 0.5144 | 0.5193 | -0.0049 | -0.0135 | 0.0031 | 5 | 0.3223 | 0.4000 | 1.0000 |
| H9 | val_macro_f1 | iwtrain_hsboa | iw_hsboa | two-sided | 10 | 0.6320 | 0.6456 | -0.0136 | -0.0223 | -0.0047 | 3 | 0.0488 | 0.2300 | 0.2930 |
| H9 | val_macro_f1_weighted | iwtrain_hsboa | iw_hsboa | two-sided | 10 | 0.6776 | 0.6852 | -0.0076 | -0.0171 | 0.0013 | 4 | 0.2754 | 0.3600 | 1.0000 |
| H10 | test_macro_f1 | iw_hsboa | default | greater | 10 | 0.5193 | 0.5345 | -0.0152 | -0.0220 | -0.0080 | 2 | 0.9971 | 0.1100 | 1.0000 |
| H10 | val_macro_f1 | iw_hsboa | default | greater | 10 | 0.6456 | 0.6477 | -0.0021 | -0.0057 | 0.0020 | 4 | 0.8389 | 0.4400 | 1.0000 |
| H10 | val_macro_f1_weighted | iw_hsboa | default | greater | 10 | 0.6852 | 0.6825 | 0.0027 | -0.0033 | 0.0100 | 6 | 0.3125 | 0.5500 | 1.0000 |

