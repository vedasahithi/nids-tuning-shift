# Macro-F1 bounds for classifiers of the 42 features

majority = group-majority labels; achievable = best assignment found (lower bound on the true maximum); upper = sum of exact per-class maxima / K (upper bound).

               set   rows  groups  majority  achievable  upper
training partition 175341  101040    0.7890      0.8029 0.8087
    test partition  82332   53946    0.7636      0.7849 0.7877
     both (merged) 257673  153684    0.7735      0.7833 0.7846

               set          class  support  majority_f1  achievable_f1  upper_f1
training partition         Normal    56000       0.9971         0.9971    0.9971
training partition        Generic    40000       0.9960         0.9960    0.9960
training partition       Exploits    33393       0.8142         0.8092    0.8183
training partition        Fuzzers    18184       0.9466         0.9470    0.9472
training partition            DoS    12264       0.4395         0.5536    0.5655
training partition Reconnaissance    10491       0.9017         0.9022    0.9023
training partition       Analysis     2000       0.4731         0.4404    0.4776
training partition       Backdoor     1746       0.3773         0.4334    0.4334
training partition      Shellcode     1133       0.9973         0.9973    0.9974
training partition          Worms      130       0.9474         0.9524    0.9524
    test partition         Normal    37000       0.9999         0.9999    0.9999
    test partition        Generic    18871       0.9991         0.9992    0.9992
    test partition       Exploits    11132       0.8054         0.8439    0.8445
    test partition        Fuzzers     6062       0.8843         0.8850    0.8850
    test partition            DoS     4089       0.6760         0.6789    0.6794
    test partition Reconnaissance     3496       0.9480         0.9480    0.9483
    test partition       Analysis      677       0.1578         0.2497    0.2623
    test partition       Backdoor      583       0.1781         0.2567    0.2706
    test partition      Shellcode      378       0.9987         0.9987    0.9987
    test partition          Worms       44       0.9885         0.9888    0.9888
     both (merged)         Normal    93000       0.9965         0.9964    0.9965
     both (merged)        Generic    58871       0.9968         0.9968    0.9968
     both (merged)       Exploits    44525       0.8106         0.8155    0.8222
     both (merged)        Fuzzers    24246       0.9241         0.9250    0.9251
     both (merged)            DoS    16353       0.5054         0.5763    0.5818
     both (merged) Reconnaissance    13987       0.9133         0.9138    0.9139
     both (merged)       Analysis     2677       0.3741         0.3764    0.3764
     both (merged)       Backdoor     2329       0.2626         0.2777    0.2778
     both (merged)      Shellcode     1511       0.9967         0.9967    0.9967
     both (merged)          Worms      174       0.9550         0.9587    0.9587
