# Tuning ablation: hyper-parameter-only search, defaults as first candidate, full-data fitness

Arms minus default (paired by seed). Decision rule in the script header.

       metric    arm  n_seeds  arm_mean  default_mean  mean_diff  ci95_low  ci95_high  seeds_arm_better  p_two_sided  equiv_within_0.005  equiv_within_0.01  p_holm
test_macro_f1 random       10    0.5258        0.5241     0.0017   -0.0023     0.0054                 6       0.4316                True               True  0.4316
test_macro_f1   sboa       10    0.5275        0.5241     0.0034    0.0010     0.0059                 6       0.0840               False               True  0.1680
 val_macro_f1 random       10    0.6457        0.6447     0.0011   -0.0000     0.0026                 6       0.1172                True               True  0.1172
 val_macro_f1   sboa       10    0.6488        0.6447     0.0041    0.0021     0.0065                 8       0.0078               False               True  0.0156

Distinct trainings and minutes per run:

         distinct_trainings  wall_min
arm                                  
default                 1.0       1.5
random                 60.0      96.1
sboa                   48.5      87.2
