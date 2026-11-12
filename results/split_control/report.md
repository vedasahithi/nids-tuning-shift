# Split control (lgbm, wall 4.5 min)

                     mean     std
settings split                   
default  G         0.6237  0.0039
         P         0.6150  0.0061
         R         0.6246  0.0030
         official  0.5345  0.0057
e2_tuned G         0.6189  0.0068
         P         0.6112  0.0097
         R         0.6207  0.0067
         official  0.5291  0.0083

Split statistics (mean over seeds):

           n_train   n_test  test_rows_with_train_twin  test_rows_twin_other_label  class_mix_tv_distance
split                                                                                                    
G         175378.6  82294.4                     0.0000                      0.0000                 0.0018
P         175340.5  78777.2                     0.0000                      0.0000                 0.1365
R         175341.0  82332.0                     0.4436                      0.0989                 0.0000
official  175341.0  82332.0                     0.1037                      0.0574                 0.1311

Decomposition (paired by seed, 95% bootstrap CI):

settings                       component   mean  ci95_low  ci95_high
 default     total_gain_R_minus_official 0.0901    0.0875     0.0932
 default    duplicate_exposure_R_minus_G 0.0009   -0.0023     0.0042
 default             class_mix_G_minus_P 0.0086    0.0041     0.0132
 default residual_shift_P_minus_official 0.0806    0.0768     0.0850
e2_tuned     total_gain_R_minus_official 0.0916    0.0867     0.0964
e2_tuned    duplicate_exposure_R_minus_G 0.0019   -0.0016     0.0049
e2_tuned             class_mix_G_minus_P 0.0077    0.0029     0.0129
e2_tuned residual_shift_P_minus_official 0.0821    0.0775     0.0862
