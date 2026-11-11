# Measured development benchmark

Artifact: `development-1500-42-20261111T004020229446Z`.

This is a real-data development experiment, not a full-dataset accuracy claim.

Population: 25,458 users / 58,431 normalized reviews; 19,958 training users / 34,904 positive training reviews.
Local catalog: 32,618; eligible ranking universe: 1,500 games selected only from training-positive counts.
Elapsed: 30.9 seconds. Selected by validation: **popularity**.

## Protocol

full training-user-selected eligible item universe, no sampled negatives; all support games excluded; other unobserved items are ranking distractors, not explicit dislikes

pooled validation held-out users support lengths 1,3,5 NDCG@20; same user may have multiple support queries

User assignment: seeded SHA-256 buckets, 80% population training / 10% validation / 10% test. Final positive of training users with >=3 positives removed for a separate warm fold-in test before item selection. Cold-test users never contribute training interactions, item popularity, CF weights or latent factors. Support is the chronological prefix of explicit reviews, including signed negatives; target is remaining positive reviews. Rows with unknown timestamps sort first. Cohorts require an unseen positive target. All algorithms share each cohort and candidate list. Cohorts may contain different users. Maximum first sorted users per cohort is configured; it is not random sampling.

Recall/NDCG/MAP at 10 and 20, explicit-negative fraction, novelty, metadata diversity and long-tail recall plus bootstrap intervals are in `data/benchmark.json`. Confidence intervals are descriptive, not evidence that any model wins statistically.

## Cold test NDCG@20 (Recall@20)

| Support | Queries | popularity | itemknn | ease | svd | content | hybrid |
|---|---|---|---|---|---|---|---|
| 0 | 300 | 0.2510 (0.4021) | 0.2510 (0.4021) | 0.2510 (0.4021) | 0.2510 (0.4021) | 0.2510 (0.4021) | 0.2510 (0.4021) |
| 1 | 300 | 0.2024 (0.3344) | 0.1168 (0.1898) | 0.1491 (0.2336) | 0.0371 (0.0836) | 0.0105 (0.0206) | 0.1357 (0.1918) |
| 3 | 300 | 0.2116 (0.3172) | 0.1774 (0.2761) | 0.1915 (0.2821) | 0.0410 (0.0909) | 0.0123 (0.0245) | 0.1776 (0.2564) |
| 5 | 158 | 0.1940 (0.3031) | 0.1702 (0.2655) | 0.1785 (0.2649) | 0.0304 (0.0664) | 0.0073 (0.0164) | 0.1578 (0.2195) |
| 10 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| 20 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| 50 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| 100 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

## Validation

| Algorithm | NDCG@20 | Recall@20 |
|---|---:|---:|
| popularity | 0.1931 | 0.3228 |
| itemknn | 0.1403 | 0.2388 |
| ease | 0.1672 | 0.2876 |
| svd | 0.0395 | 0.0804 |
| content | 0.0151 | 0.0316 |
| hybrid | 0.1480 | 0.2261 |

## Limits

- Historical Steam review-selection bias; not a probability of real-world enjoyment
- Item-universe restriction omits cold and low-count items
- Metadata snapshot is static and may postdate historical interactions
- Warm holdout tests item fold-in; no per-user stored latent vectors
- Single seeded split; no significance claim from overlapping confidence intervals
- No global chronological temporal benchmark yet
- No semantic embedding comparison yet

No numerical-rating RMSE/MAE: source feedback is binary recommendation, not a valid star-rating scale.

Model weights trained on UCSD research data must remain local until distribution rights are clarified. Source code license does not cover data or derived weights.

Dataset SHA-256: `3dee0cbe55b3800acf70f5279f98fd5858062e2702f67619327ec13ae573f22c`
Split SHA-256: `722aa391da52f8beaa67c8ba92dadbe450fa9af39d34b29da13c1204e5aa0475`

Reproduce: `gamecompass benchmark --max-items 1500 --max-users 500 --seed 42` (use exact recorded sizes if changed).
