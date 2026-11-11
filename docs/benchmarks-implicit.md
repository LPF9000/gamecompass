# Implicit-only personalization adapter

entire validation/test users held out from CF training; reviewed game IDs and source hours revealed without support recommendation labels; remaining positive reviews as targets; full eligible item ranking; support excluded

This experiment hides support labels and is different from explicit enjoyment personalization. It does not equate ownership or hours with liking a game.

Validation compares uniform observed-game weights with log1p(hours), across popularity, item-kNN, EASE, SVD, content and the validated hybrid. Select a weighting and algorithm per history bucket only when at least 30 validation queries exist; otherwise use the closest validated lower bucket or pooled winner.

| History | Test queries | Adapter | NDCG@20 | Recall@20 | Popularity NDCG@20 | Explicit reference NDCG@20 |
|---|---:|---|---:|---:|---:|---:|
| 0 | 300 | uniform + popularity | 0.0616 | 0.1422 | 0.0616 | 0.0616 |
| 1 | 300 | uniform + hybrid | 0.1317 | 0.2241 | 0.0643 | 0.1231 |
| 3 | 300 | uniform + ease | 0.1036 | 0.1601 | 0.0502 | 0.0960 |
| 5 | 300 | log1p_hours + hybrid | 0.1200 | 0.1786 | 0.0502 | 0.1115 |
| 10 | 300 | log1p_hours + ease | 0.1174 | 0.1495 | 0.0398 | 0.1109 |
| 20 | 300 | log1p_hours + ease | 0.1259 | 0.1362 | 0.0389 | 0.1153 |
| 50 | 34 | log1p_hours + ease | 0.1483 | 0.1167 | 0.0563 | 0.1602 |
| 100 | 8 | log1p_hours + ease | 0.1823 | 0.1283 | 0.0483 | 0.1869 |

## Limits

- Reviewed games are an exposure-selected proxy for a game library; ownership-only real-world quality is unverified
- Snapshot playtime may include activity after historical review date; this is not a temporal implicit benchmark
- Playtime and ownership are weaker signals than explicit enjoyment; no label conversion in persistent data
- Mixed explicit-plus-implicit weighting has not been validated

Full validation/test aggregates and confidence intervals: implicit-benchmark-results.json. No raw histories are included in this report.
