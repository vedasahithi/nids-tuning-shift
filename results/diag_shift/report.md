# Shift diagnostic (test features, no test labels)

Wall time 3.1 min.

## Decision rule

- S1 covariate shift: class-mix-matched train-vs-test AUC 0.8969 (>= 0.7), control AUC 0.5013 (<= 0.55) -> **met** (unmatched train-vs-test AUC 0.8923; the difference is what the class mix alone explains)
- S2 concentrated: greedy AUC_0 0.8954, target <= 0.6977 within 10 removed -> **not met**
- Outcome: **S1 only: covariate shift spread over many features -> contributions 1 + 2, DANN / importance weighting as the adaptation comparison.**

## Domain AUC by comparison and seed

| comparison | seed | auc |
|---|---|---|
| train_vs_test | 0 | 0.8926 |
| train_vs_test | 1 | 0.8920 |
| train_vs_test | 2 | 0.8923 |
| matched_vs_test | 0 | 0.8970 |
| matched_vs_test | 1 | 0.8971 |
| matched_vs_test | 2 | 0.8968 |
| control | 0 | 0.4978 |
| control | 1 | 0.5038 |
| control | 2 | 0.5023 |

## Greedy removal (features the domain classifier relies on most)

| removed | auc | dropped | importance |
|---|---|---|---|
| 0 | 0.8954 | ct_dst_src_ltm | 0.2390 |
| 1 | 0.8738 | ct_dst_sport_ltm | 0.0795 |
| 2 | 0.8643 | ct_src_dport_ltm | 0.0893 |
| 3 | 0.8628 | ct_dst_ltm | 0.0650 |
| 4 | 0.8579 | ct_src_ltm | 0.0924 |
| 5 | 0.8291 | ct_srv_dst | 0.1017 |
| 6 | 0.8254 | ct_srv_src | 0.1080 |
| 7 | 0.7831 | dur | 0.0830 |
| 8 | 0.7788 | sinpkt | 0.0484 |
| 9 | 0.7715 | rate | 0.0793 |
| 10 | 0.7695 | sload | 0.0820 |
| 11 | 0.7276 | dmean | 0.0458 |
| 12 | 0.7238 | dbytes | 0.0537 |
| 13 | 0.7174 | sttl | 0.0512 |
| 14 | 0.7169 | ct_state_ttl | 0.0928 |
| 15 | 0.7161 |  |  |

## Univariate drift, top 20 features (test vs training partition)

| feature | auc | ks | psi | mean_ref | mean_other | median_ref | median_other |
|---|---|---|---|---|---|---|---|
| ct_dst_src_ltm | 0.5535 | 0.1003 | 0.0775 | 8.7705 | 7.4564 | 3.0000 | 3.0000 |
| sttl | 0.5533 | 0.1246 | 0.0853 | 161.2297 | 180.9677 | 254.0000 | 254.0000 |
| ct_state_ttl | 0.5459 | 0.1166 | 0.0849 | 1.1704 | 1.3693 | 1.0000 | 1.0000 |
| ackdat | 0.5408 | 0.1235 | 0.0614 | 0.0178 | 0.0267 | 0.0000 | 0.0001 |
| tcprtt | 0.5376 | 0.1066 | 0.0553 | 0.0367 | 0.0559 | 0.0000 | 0.0006 |
| synack | 0.5365 | 0.1060 | 0.0544 | 0.0189 | 0.0293 | 0.0000 | 0.0004 |
| ct_dst_ltm | 0.5344 | 0.0721 | 0.0630 | 6.3716 | 5.7449 | 3.0000 | 2.0000 |
| dload | 0.5321 | 0.0962 | 0.0531 | 936133.2317 | 630546.9590 | 2815.8466 | 2112.9514 |
| ct_src_ltm | 0.5310 | 0.0670 | 0.1088 | 7.0488 | 6.4684 | 3.0000 | 3.0000 |
| dttl | 0.5299 | 0.1103 | 0.0910 | 71.2946 | 95.7130 | 29.0000 | 29.0000 |
| sinpkt | 0.5288 | 0.0824 | 0.0338 | 1333.4618 | 755.3943 | 0.4243 | 0.5579 |
| dmean | 0.5249 | 0.0810 | 0.0514 | 145.0854 | 116.2751 | 44.0000 | 44.0000 |
| sjit | 0.5244 | 0.0833 | 0.0462 | 4722.5420 | 6363.0751 | 2.0804 | 17.6239 |
| service | 0.5233 | 0.0457 | 0.0087 | 1.6219 | 1.4217 | 0.0000 | 0.0000 |
| dur | 0.5231 | 0.0724 | 0.0615 | 1.1946 | 1.0068 | 0.0046 | 0.0141 |
| rate | 0.5225 | 0.0725 | 0.1131 | 85198.8021 | 82410.8867 | 2921.1294 | 2650.1767 |
| swin | 0.5212 | 0.0423 | 0.0072 | 122.6661 | 133.4591 | 0.0000 | 255.0000 |
| dinpkt | 0.5192 | 0.0714 | 0.0350 | 89.2985 | 121.7013 | 0.0090 | 0.0100 |
| dbytes | 0.5182 | 0.0690 | 0.0296 | 17646.1250 | 13233.7856 | 178.0000 | 178.0000 |
| sload | 0.5174 | 0.0628 | 0.0397 | 63681276.6153 | 64549016.9141 | 692050.6250 | 577003.2188 |

## Duplicates

```
{
  "rows": 175341,
  "unique_rows": 101040,
  "dup_row_share": 0.4793459601576357,
  "conflict_row_share": 0.17410645542115077,
  "other_rows": 82332,
  "other_seen_share": 0.10373852208132925,
  "other_seen_conflicting_share": 0.03230821551766021,
  "other_dup_row_share": 0.4215128989943157
}
```
