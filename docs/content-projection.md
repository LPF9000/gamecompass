# Supervised content-to-collaborative cold-item experiment

The experiment uses the same 279 games selected by fixed item hashes and the same held-out test-user cohorts as the cold-item evidence ablation. A fresh SVD is fitted on warm-item interactions from training users. Ridge regression maps metadata TF-IDF to normalized warm latent factors and predicts normalized cold latent factors. Ranking uses signed warm-user aggregation over every cold candidate. Alpha is selected using validation users only.

Fresh collaborative fit: 280,689 training users, 527,228 positive warm-item interactions, 1221 warm items, 279 cold items wholly excluded. Latent dimensions: 64. Elapsed: 23.3s.

No population validation/test user or cold-item interaction enters SVD or ridge fitting. The prior production model supplies only unsupervised metadata features and item IDs; its CF weights and popularity counts are never used.

## Validation-only alpha selection

| Ridge alpha | Pooled validation NDCG@20 |
|---|---:|
| 0.1 | 0.0304 |
| 1.0 | 0.0321 |
| 10.0 | 0.0343 |
| 100.0 | 0.0354 |

Selected alpha: **100.0**. Plain TF-IDF validation NDCG@20: 0.0268.

## Untouched test users

| Warm support | Queries | Projection NDCG@20 (95% CI) | Projection Recall@20 | TF-IDF NDCG@20 | Zero-evidence NDCG@20 |
|---|---:|---|---:|---:|---:|
| 1 | 300 | 0.0294 (0.0173–0.0435) | 0.0609 | 0.0220 | 0.0207 |
| 3 | 300 | 0.0377 (0.0266–0.0508) | 0.0805 | 0.0349 | 0.0273 |
| 5 | 300 | 0.0382 (0.0286–0.0472) | 0.0889 | 0.0292 | 0.0202 |

## Limits

- Candidate scope was selected from train-user-positive counts before item withholding, so cold candidates are historically observed popular games, not actual new releases
- Metadata snapshot may postdate review times; this is not a release-date temporal experiment
- Queries are conditional on a later positive cold target and first sorted capped users; not all-user satisfaction estimates
- Single seeded split with descriptive user-bootstrap intervals; pooled validation queries repeat some users
- SVD normalization and ridge alpha grid are fixed modeling choices; no evidence of universal optimality
- Zero evidence baseline is a canonical tie ordering; other constant tie orders can change that baseline

not integrated; experiment does not change model weights or production pointer

Full aggregate results: content-projection-results.json.

Reproduce: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/benchmark_content_projection.py --data-dir data --max-users 300 --seed 42`.
