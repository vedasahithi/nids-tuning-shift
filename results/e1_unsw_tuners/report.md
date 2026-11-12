# E1 tuner comparison: summary

Finished runs per tuner: {'hsboa': 10, 'random': 10} (planned 10 seeds). Budget (evaluations per tuner, matched on distinct trainings) {'hsboa': 48, 'random': 39} (pop 8), 100 epochs per trial, 200 final. Wilcoxon signed-rank paired by seed; bootstrap 95% CI of the mean difference; Holm over the primary metrics (validation and test macro-F1).

## Per run

| tuner | seed | search_val_macro_f1 | val_macro_f1 | test_macro_f1 | val_f1_Shellcode | val_f1_Worms | wall_min |
|---|---|---|---|---|---|---|---|
| hsboa | 0 | 0.4236 | 0.4606 | 0.3620 | 0.2937 | 0.0000 | 212.7470 |
| random | 0 | 0.4272 | 0.4597 | 0.3656 | 0.2585 | 0.0198 | 249.7354 |
| hsboa | 1 | 0.4367 | 0.4380 | 0.3587 | 0.1196 | 0.0000 | 200.0182 |
| random | 1 | 0.4399 | 0.4575 | 0.3623 | 0.2187 | 0.0417 | 271.2332 |
| hsboa | 2 | 0.4082 | 0.4428 | 0.3604 | 0.1075 | 0.0000 | 226.1063 |
| random | 2 | 0.4252 | 0.4599 | 0.3660 | 0.2068 | 0.0849 | 227.5769 |
| hsboa | 3 | 0.4032 | 0.3717 | 0.2909 | 0.0000 | 0.0000 | 224.8959 |
| random | 3 | 0.4156 | 0.4364 | 0.3373 | 0.1254 | 0.0000 | 267.4499 |
| hsboa | 4 | 0.4497 | 0.4457 | 0.3547 | 0.1873 | 0.0222 | 223.3397 |
| random | 4 | 0.4401 | 0.4346 | 0.3340 | 0.1009 | 0.0519 | 241.7164 |
| hsboa | 5 | 0.4514 | 0.4687 | 0.3890 | 0.2981 | 0.0000 | 303.2480 |
| random | 5 | 0.4304 | 0.4528 | 0.3722 | 0.2079 | 0.0000 | 249.3140 |
| hsboa | 6 | 0.4170 | 0.4267 | 0.3413 | 0.1667 | 0.0000 | 325.4572 |
| random | 6 | 0.4093 | 0.4579 | 0.3767 | 0.2156 | 0.0569 | 257.4708 |
| hsboa | 7 | 0.4239 | 0.4635 | 0.3829 | 0.2816 | 0.0000 | 294.0993 |
| random | 7 | 0.4317 | 0.4554 | 0.3663 | 0.2176 | 0.0000 | 262.5873 |
| hsboa | 8 | 0.3757 | 0.3950 | 0.3270 | 0.0000 | 0.0000 | 116.3881 |
| random | 8 | 0.4271 | 0.4378 | 0.3694 | 0.1208 | 0.0000 | 123.9485 |
| hsboa | 9 | 0.4004 | 0.4094 | 0.3121 | 0.0000 | 0.0000 | 98.3332 |
| random | 9 | 0.4160 | 0.4422 | 0.3628 | 0.1222 | 0.0488 | 106.4412 |

## Paired comparisons

| metric | a | b | n_seeds | mean_a | mean_b | mean_diff | ci95_low | ci95_high | p | a12_a_better | p_holm_primary |
|---|---|---|---|---|---|---|---|---|---|---|---|
| search_val_macro_f1 | hsboa | random | 10 | 0.4190 | 0.4263 | -0.0073 | -0.0193 | 0.0035 | 0.3223 | 0.3700 | nan |
| val_accuracy | hsboa | random | 10 | 0.7360 | 0.7359 | 0.0001 | -0.0082 | 0.0084 | 1.0000 | 0.5000 | nan |
| val_balanced_accuracy | hsboa | random | 10 | 0.4493 | 0.4809 | -0.0315 | -0.0512 | -0.0115 | 0.0371 | 0.2400 | nan |
| val_macro_f1 | hsboa | random | 10 | 0.4322 | 0.4494 | -0.0172 | -0.0328 | -0.0024 | 0.0840 | 0.4100 | 0.1680 |
| val_mcc | hsboa | random | 10 | 0.6769 | 0.6798 | -0.0029 | -0.0126 | 0.0058 | 0.6953 | 0.5300 | nan |
| val_f1_Normal | hsboa | random | 10 | 0.8693 | 0.8662 | 0.0031 | -0.0005 | 0.0068 | 0.2324 | 0.6000 | nan |
| val_f1_Generic | hsboa | random | 10 | 0.9759 | 0.9801 | -0.0042 | -0.0087 | -0.0002 | 0.3750 | 0.4300 | nan |
| val_f1_Exploits | hsboa | random | 10 | 0.6174 | 0.6071 | 0.0102 | -0.0051 | 0.0259 | 0.2754 | 0.5900 | nan |
| val_f1_Fuzzers | hsboa | random | 10 | 0.5756 | 0.5962 | -0.0206 | -0.0318 | -0.0100 | 0.0098 | 0.2100 | nan |
| val_f1_DoS | hsboa | random | 10 | 0.3896 | 0.4134 | -0.0238 | -0.0763 | 0.0138 | 0.3223 | 0.3000 | nan |
| val_f1_Reconnaissance | hsboa | random | 10 | 0.5797 | 0.6063 | -0.0266 | -0.0736 | 0.0067 | 0.3223 | 0.4200 | nan |
| val_f1_Analysis | hsboa | random | 10 | 0.1607 | 0.2053 | -0.0447 | -0.0835 | -0.0133 | 0.0195 | 0.3800 | nan |
| val_f1_Backdoor | hsboa | random | 10 | 0.0060 | 0.0096 | -0.0036 | -0.0200 | 0.0109 | 1.0000 | 0.4950 | nan |
| val_f1_Shellcode | hsboa | random | 10 | 0.1454 | 0.1794 | -0.0340 | -0.0865 | 0.0202 | 0.1602 | 0.4000 | nan |
| val_f1_Worms | hsboa | random | 10 | 0.0022 | 0.0304 | -0.0282 | -0.0460 | -0.0116 | 0.0312 | 0.2300 | nan |
| test_accuracy | hsboa | random | 10 | 0.6175 | 0.6164 | 0.0011 | -0.0140 | 0.0163 | 1.0000 | 0.5400 | nan |
| test_balanced_accuracy | hsboa | random | 10 | 0.3936 | 0.4246 | -0.0310 | -0.0536 | -0.0079 | 0.0645 | 0.2400 | nan |
| test_macro_f1 | hsboa | random | 10 | 0.3479 | 0.3613 | -0.0133 | -0.0296 | 0.0027 | 0.2324 | 0.3000 | 0.2324 |
| test_mcc | hsboa | random | 10 | 0.5423 | 0.5467 | -0.0044 | -0.0192 | 0.0099 | 0.6250 | 0.4600 | nan |
| test_f1_Normal | hsboa | random | 10 | 0.7038 | 0.6893 | 0.0144 | 0.0010 | 0.0311 | 0.0645 | 0.6900 | nan |
| test_f1_Generic | hsboa | random | 10 | 0.8652 | 0.8860 | -0.0208 | -0.0499 | 0.0075 | 0.2754 | 0.4400 | nan |
| test_f1_Exploits | hsboa | random | 10 | 0.5943 | 0.5965 | -0.0022 | -0.0194 | 0.0136 | 1.0000 | 0.4900 | nan |
| test_f1_Fuzzers | hsboa | random | 10 | 0.3150 | 0.3279 | -0.0129 | -0.0205 | -0.0046 | 0.0273 | 0.3400 | nan |
| test_f1_DoS | hsboa | random | 10 | 0.3216 | 0.3508 | -0.0291 | -0.0797 | 0.0184 | 0.4316 | 0.4000 | nan |
| test_f1_Reconnaissance | hsboa | random | 10 | 0.5455 | 0.5618 | -0.0162 | -0.0568 | 0.0221 | 0.3750 | 0.4500 | nan |
| test_f1_Analysis | hsboa | random | 10 | 0.0148 | 0.0266 | -0.0118 | -0.0236 | 0.0006 | 0.1172 | 0.2700 | nan |
| test_f1_Backdoor | hsboa | random | 10 | 0.0036 | 0.0045 | -0.0008 | -0.0092 | 0.0069 | 0.6250 | 0.4200 | nan |
| test_f1_Shellcode | hsboa | random | 10 | 0.1112 | 0.1487 | -0.0375 | -0.0858 | 0.0121 | 0.1602 | 0.3800 | nan |
| test_f1_Worms | hsboa | random | 10 | 0.0040 | 0.0205 | -0.0165 | -0.0296 | -0.0040 | 0.0625 | 0.3100 | nan |

