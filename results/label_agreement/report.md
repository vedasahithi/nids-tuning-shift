# Cross-partition label agreement (identical 42-feature records)

                                                        quantity  value  ci_lo  ci_hi
                       test rows with a training twin (coverage) 0.1037    NaN    NaN
                  test groups with a training twin / test groups 0.0241    NaN    NaN
      cross-partition pairwise disagreement (test vs train twin) 0.5564 0.5130 0.5997
        within-train pairwise disagreement (all matched classes) 0.2516 0.2398 0.2634
within-train disagreement standardised to matched test class mix 0.3422 0.3174 0.3657
      excess disagreement: cross minus standardised within-train 0.2142 0.1909 0.2375

Per class:

         class  test_rows  test_rows_with_train_twin  coverage  cross_disagree  within_train_rows  within_disagree  diff_cross_minus_within  diff_ci_lo  diff_ci_hi
        Normal    37000.0                     1472.0    0.0398          0.2882             6047.0           0.0461                   0.2421      0.1989      0.2894
       Generic    18871.0                     1722.0    0.0913          0.0163            39169.0           0.0080                   0.0082      0.0020      0.0152
      Exploits    11132.0                     1802.0    0.1619          0.9561            16596.0           0.4500                   0.5061      0.4819      0.5274
       Fuzzers     6062.0                      261.0    0.0431          0.9573             4649.0           0.3865                   0.5707      0.5392      0.5986
           DoS     4089.0                     2004.0    0.4901          0.9682             9822.0           0.6842                   0.2840      0.2692      0.2978
Reconnaissance     3496.0                     1138.0    0.3255          0.2268             4617.0           0.3950                  -0.1682     -0.2348     -0.1124
      Analysis      677.0                      125.0    0.1846          0.9931             1708.0           0.8266                   0.1665      0.1387      0.1950
      Backdoor      583.0                        4.0    0.0069          0.9444             1346.0           0.9745                  -0.0301     -0.9807     -0.0225
     Shellcode      378.0                       13.0    0.0344          0.0385               82.0           0.0732                  -0.0347     -0.1111      0.0434
         Worms       44.0                        0.0    0.0000          0.0000               13.0           0.8462                  -0.8462     -1.0000     -0.7500
