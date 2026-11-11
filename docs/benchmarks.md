# Measured development benchmark

Artifact: `development-1500-42-20261111T004254213166Z`.

This is a real-data development experiment, not a full-dataset accuracy claim.

Dataset: antonkozyriev/game-recommendations-on-steam; version 28. Cohort: numeric source user_id % 32 == 0; complete selected histories.

Population: 430,864 users / 1,295,681 normalized reviews; 321,213 training users / 658,217 positive training reviews.
Local catalog: 50,872; eligible ranking universe: 1,500 games selected only from training-positive counts.
Elapsed: 102.0 seconds. Selected by validation: **hybrid**.

## Protocol

full training-user-selected eligible item universe, no sampled negatives; all support games excluded; other unobserved items are ranking distractors, not explicit dislikes

pooled validation held-out users support lengths 1,3,5 NDCG@20; same user may have multiple support queries

User assignment: seeded SHA-256 buckets, 80% population training / 10% validation / 10% test. Final positive of training users with >=3 positives removed for a separate warm fold-in test before item selection. Cold-test users never contribute training interactions, item popularity, CF weights or latent factors. Support is the chronological prefix of explicit reviews, including signed negatives; target is remaining positive reviews. History lengths count reviews only within the eligible item universe; interactions outside that scope are not evaluated. Rows with unknown timestamps sort first. Cohorts require an unseen positive target. All algorithms share each cohort and candidate list. Cohorts may contain different users. Maximum first sorted users per cohort is configured; it is not random sampling.

Recall/NDCG/MAP at 10 and 20, explicit-negative fraction, novelty, metadata diversity and long-tail recall plus bootstrap intervals are in `data/benchmark.json`. Confidence intervals are descriptive, not evidence that any model wins statistically.

## Cold test NDCG@20 (Recall@20)

| Support | Queries | popularity | itemknn | ease | svd | content | hybrid |
|---|---|---|---|---|---|---|---|
| 0 | 300 | 0.0616 (0.1422) | 0.0616 (0.1422) | 0.0616 (0.1422) | 0.0616 (0.1422) | 0.0616 (0.1422) | 0.0616 (0.1422) |
| 1 | 300 | 0.0643 (0.1200) | 0.1119 (0.1919) | 0.1195 (0.1929) | 0.0820 (0.1492) | 0.0109 (0.0131) | 0.1231 (0.2060) |
| 3 | 300 | 0.0502 (0.0868) | 0.0883 (0.1441) | 0.0960 (0.1472) | 0.0647 (0.0982) | 0.0170 (0.0228) | 0.0998 (0.1556) |
| 5 | 300 | 0.0502 (0.0961) | 0.0965 (0.1612) | 0.1107 (0.1705) | 0.0798 (0.1081) | 0.0121 (0.0159) | 0.1115 (0.1711) |
| 10 | 300 | 0.0398 (0.0616) | 0.0941 (0.1134) | 0.1109 (0.1348) | 0.0625 (0.0806) | 0.0257 (0.0308) | 0.1080 (0.1312) |
| 20 | 300 | 0.0389 (0.0457) | 0.0968 (0.0992) | 0.1157 (0.1198) | 0.0925 (0.1010) | 0.0235 (0.0194) | 0.1153 (0.1203) |
| 50 | 34 | 0.0563 (0.0449) | 0.1267 (0.0752) | 0.1602 (0.1275) | 0.1482 (0.1118) | 0.0421 (0.0228) | 0.1551 (0.1242) |
| 100 | 8 | 0.0483 (0.0203) | 0.1798 (0.1252) | 0.1869 (0.1513) | 0.1899 (0.1315) | 0.0908 (0.1201) | 0.1839 (0.1401) |

## Validation

| Algorithm | NDCG@20 | Recall@20 |
|---|---:|---:|
| popularity | 0.0468 | 0.0809 |
| itemknn | 0.0807 | 0.1332 |
| ease | 0.0953 | 0.1526 |
| svd | 0.0574 | 0.1063 |
| content | 0.0112 | 0.0172 |
| hybrid | 0.0960 | 0.1564 |

## Validation-selected routing

The pooled winner is used when no eligible history bucket exists. Otherwise the algorithm selected on at least 30 validation queries for the nearest lower history bucket is used; test results do not choose routes.

| Minimum explicit history | Algorithm |
|---|---|
| 0 | popularity |
| 1 | hybrid |
| 3 | ease |
| 5 | hybrid |
| 10 | ease |
| 20 | hybrid |
| 50 | ease |

## Production routing on untouched test users

| Support | Algorithm | Queries | NDCG@20 | Recall@20 |
|---|---|---:|---:|---:|
| 0 | popularity | 300 | 0.0616 | 0.1422 |
| 1 | hybrid | 300 | 0.1231 | 0.2060 |
| 3 | ease | 300 | 0.0960 | 0.1472 |
| 5 | hybrid | 300 | 0.1115 | 0.1711 |
| 10 | ease | 300 | 0.1109 | 0.1348 |
| 20 | hybrid | 300 | 0.1153 | 0.1203 |
| 50 | ease | 34 | 0.1602 | 0.1275 |
| 100 | ease | 8 | 0.1869 | 0.1513 |

## Limits

- Historical Steam review-selection bias; not a probability of real-world enjoyment
- Item-universe restriction omits cold and low-count items
- Metadata snapshot is static and may postdate historical interactions
- Warm holdout tests item fold-in; no per-user stored latent vectors
- Single seeded split; no significance claim from overlapping confidence intervals
- No semantic embedding comparison yet

No numerical-rating RMSE/MAE: source feedback is binary recommendation, not a valid star-rating scale.

Source code licensing is separate from data and trained weights. Consult the dataset provenance and licensing report before distributing artifacts.

Dataset SHA-256: `e873ed8dc766549b66d69496a6580d38242fa84786f6ce029609241b94e0a76f`
Split SHA-256: `d73cdf559e70d7eb1c3b2aa7cf43b3985b4cb77f1b031a3a5d04a41d43db0ae0`

Reproduce: `gamecompass benchmark --max-items 1500 --max-users 300 --seed 42`.
