# Unsearched full-feature baseline

       arm part  n_seeds  macro_f1     sd  reference_macro_f1
 hgb_plain  val       10    0.4253 0.1055              0.6447
 hgb_plain test       10    0.3548 0.0884              0.5241
lgbm_plain  val       10    0.5398 0.0082              0.6477
lgbm_plain test       10    0.4427 0.0084              0.5345

Plain minus the paper's untuned arm (20 features, alpha 0.5):

       arm part             reference    diff  ci95_lo  ci95_hi  seeds_better  wilcoxon_p
 hgb_plain  val e1b_unsw_gbdt/default -0.2194  -0.2911  -0.1684             0       0.002
 hgb_plain test e1b_unsw_gbdt/default -0.1694  -0.2281  -0.1296             0       0.002
lgbm_plain  val e6_unsw_shift/default -0.1079  -0.1151  -0.1008             0       0.002
lgbm_plain test e6_unsw_shift/default -0.0918  -0.0985  -0.0851             0       0.002