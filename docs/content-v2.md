# Structured game content v2 — validation only

The prior content feature path omitted the real Kaggle `metadata.tags` field. This version reads those tags as atomic labels, deduplicates them, and gives descriptions a separate fitted word/bigram TF-IDF vocabulary. Tags and descriptions are normalized independently; feature blend weights are selected on validation users.

Catalog: 50,872 games; 49,628 have structured tags. Fitted features: 441 atomic tags and 12,000 description terms/bigrams.

Fresh SVD fit uses 280,689 training users and 527,228 positive interactions over 1221 warm items. Every one of the 279 cold items and every validation/test user is excluded from collaborative fitting.

Selected cosine tag weight: 0.5; selected ridge tag weight: 0.75, alpha: 1.0. Default content method: **projection**. Selection uses pooled cold-item validation NDCG@20, support lengths1/3/5; no test ranking is computed.

## Cold-item validation

| Warm support | Queries | Tags NDCG@20 | Description NDCG@20 | Selected cosine NDCG@20 | Projection NDCG@20 |
|---|---:|---:|---:|---:|---:|
| 1 | 300 | 0.0222 | 0.0185 | 0.0229 | 0.0402 |
| 3 | 300 | 0.0430 | 0.0216 | 0.0465 | 0.0546 |
| 5 | 300 | 0.0482 | 0.0486 | 0.0552 | 0.0631 |

## Full-catalog validation

Candidate universe contains every cached catalog game, with support games excluded. Query targets are later positive reviews inside the bounded evidence universe. Queries are not conditioned on having a cold target. The same queries/candidates apply to every method.

| Warm support | Queries | Warm CF only NDCG@20 | Tags | Description | Selected cosine | Projection |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 0.0508 | 0.0100 | 0.0039 | 0.0047 | 0.0202 |
| 3 | 100 | 0.0409 | 0.0079 | 0.0010 | 0.0030 | 0.0101 |
| 5 | 100 | 0.0359 | 0.0153 | 0.0032 | 0.0030 | 0.0124 |

## Artifact and API

Frozen artifact: `data/content-models/content-v2-validation-20261111T011640236025Z`. Its manifest records vocabularies, feature parameters, selection, source fingerprint and component checksums. Arrays store fitted IDF, ridge coefficients and warm factors; sparse NPZ files store catalog feature blocks. No pickle or API account is needed.

`ContentModel.load(path).score(support, candidates=None, support_games=None, algorithm=None)` returns ranking scores aligned with supplied candidate game dictionaries or canonical ID strings; omitted candidates mean the cached catalog. Algorithms: `tags`, `description`, `cosine`, `projection`. Repeated known candidate sets use cached sparse/dense views. New candidate metadata is transformed with fitted vocabularies/IDF. New support metadata can be supplied through `support_games`. Unknown/missing metadata gives zero evidence.

For projection, warm games use their fitted collaborative factors; every cold/new game uses predicted factors. User representation is a normalized signed aggregation. Original production weights and pointer are not changed.

## Limits

- Validation was previously used during exploration; it is suitable for selection, not a fresh accuracy claim
- No test ranking has been run in this experiment; a reserved final test is required
- Cold-item scope is a hash-selected subset of historically observed train-user-popular1500 games, not actual new releases
- Static metadata tags/descriptions may postdate historical interactions
- Pooled validation queries repeat users across support lengths; bootstrap intervals are descriptive
- Broad-catalog targets cover only later labeled positives in the bounded1500-item evidence universe; other games are unobserved distractors
- Source is historical Steam review feedback; console coverage and future enjoyment remain unverified

Elapsed: 107.9s. Full selection grid and metrics: content-v2-validation-results.json.

Reproduce: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/train_content_v2.py --data-dir data --max-users 300 --catalog-users 100 --seed 42`.
