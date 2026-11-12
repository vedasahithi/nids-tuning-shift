# Support check (seed 0, 10,000 queries each)

           query set  records  share with exact training twin  median distance (no twin)  p90 (no twin)  p99 (no twin)  share beyond validation p99 (no twin)
validation (control)    10000                          0.4049                     0.1397         0.2794         0.4839                                 0.0101
                test    10000                          0.0843                     0.1545         0.3468         0.7358                                 0.0327

By class (records without a training twin):

         class  test records (no twin)  test median distance  validation records (no twin)  validation median distance
        Normal                    4280                0.1605                          2879                      0.1486
       Generic                    2114                0.0468                           226                      0.0000
      Exploits                    1187                0.1834                          1190                      0.1383
       Fuzzers                     691                0.2806                           843                      0.1616
           DoS                     360                0.1778                           278                      0.0000
Reconnaissance                     330                0.0977                           396                      0.0878
      Analysis                      76                0.2294                            35                      0.1406
      Backdoor                      63                0.2655                            36                      0.0361
     Shellcode                      52                0.1356                            62                      0.1379
         Worms                       4                0.1753                             6                      0.2737
