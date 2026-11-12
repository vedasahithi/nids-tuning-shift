# Paired statistics (seeds 0-9)

Contrasts: a = hgb_hsboa; difference a - b; 95% percentile bootstrap CI over seeds; two-sided exact Wilcoxon; Holm over the 16 comparisons.

hypothesis        metric         a               b  mean_diff  sd_diff  ci95_low  ci95_high      p  seeds_a_better  p_holm_h1_h4_16  p_holm_h1_h4_by_metric_8
        H1  val_macro_f1 hgb_hsboa      qrnn_hsboa     0.2000   0.0335    0.1812     0.2203 0.0020              10           0.0312                    0.0156
        H1  val_macro_f1 hgb_hsboa     qrnn_random     0.1828   0.0152    0.1738     0.1917 0.0020              10           0.0312                    0.0156
        H1  val_macro_f1 hgb_hsboa qrnn_random_d20     0.1752   0.0329    0.1579     0.1960 0.0020              10           0.0312                    0.0156
        H1  val_macro_f1 hgb_hsboa      cnn_bilstm     0.0386   0.0229    0.0251     0.0522 0.0039               9           0.0312                    0.0156
        H1  val_macro_f1 hgb_hsboa             mlp     0.0496   0.0243    0.0354     0.0636 0.0020              10           0.0312                    0.0156
        H2  val_macro_f1 hgb_hsboa      hgb_random     0.0030   0.0127   -0.0041     0.0106 0.6953               5           1.0000                    1.0000
        H3  val_macro_f1 hgb_hsboa        hgb_sboa    -0.0019   0.0063   -0.0057     0.0016 0.6602               5           1.0000                    1.0000
        H4  val_macro_f1 hgb_hsboa     hgb_default    -0.0125   0.0112   -0.0192    -0.0062 0.0098               2           0.0586                    0.0293
        H1 test_macro_f1 hgb_hsboa      qrnn_hsboa     0.1663   0.0345    0.1465     0.1863 0.0020              10           0.0312                    0.0156
        H1 test_macro_f1 hgb_hsboa     qrnn_random     0.1529   0.0231    0.1400     0.1671 0.0020              10           0.0312                    0.0156
        H1 test_macro_f1 hgb_hsboa qrnn_random_d20     0.1416   0.0354    0.1226     0.1633 0.0020              10           0.0312                    0.0156
        H1 test_macro_f1 hgb_hsboa      cnn_bilstm     0.0375   0.0215    0.0246     0.0497 0.0020              10           0.0312                    0.0156
        H1 test_macro_f1 hgb_hsboa             mlp     0.0545   0.0163    0.0459     0.0648 0.0020              10           0.0312                    0.0156
        H2 test_macro_f1 hgb_hsboa      hgb_random     0.0136   0.0207    0.0013     0.0253 0.0645               7           0.3223                    0.1934
        H3 test_macro_f1 hgb_hsboa        hgb_sboa    -0.0013   0.0052   -0.0044     0.0017 0.3477               2           1.0000                    0.3477
        H4 test_macro_f1 hgb_hsboa     hgb_default    -0.0100   0.0144   -0.0183    -0.0016 0.1055               3           0.4219                    0.2109

Equivalence (90% bootstrap CI inside +-margin; TOST p):

hypothesis        metric           b  mean_diff  ci90_low  ci90_high  equiv_within_0.005  tost_p_0.005  equiv_within_0.01  tost_p_0.01  equiv_within_0.02  tost_p_0.02
        H2  val_macro_f1  hgb_random     0.0030   -0.0030     0.0094               False        0.3129               True       0.0570               True       0.0011
        H3  val_macro_f1    hgb_sboa    -0.0019   -0.0051     0.0011               False        0.0739               True       0.0013               True       0.0000
        H4  val_macro_f1 hgb_default    -0.0125   -0.0181    -0.0071               False        0.9686              False       0.7523               True       0.0319
        H2 test_macro_f1  hgb_random     0.0136    0.0030     0.0236               False        0.8901              False       0.7041              False       0.1786
        H3 test_macro_f1    hgb_sboa    -0.0013   -0.0039     0.0013                True        0.0253               True       0.0002               True       0.0000
        H4 test_macro_f1 hgb_default    -0.0100   -0.0171    -0.0030               False        0.8478              False       0.4966               True       0.0275
