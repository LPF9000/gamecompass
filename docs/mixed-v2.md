# Mixed explicit and implicit validation experiment

**Validation only. No test-user predictions or production activation occurred.**

Source: 1,295,681 reviews / 430,864 users. SHA-256 `e873ed8dc766549b66d69496a6580d38242fa84786f6ce029609241b94e0a76f`. Base model `development-1500-42-20261111T004254213166Z`.

Original seed42 whole-user validation partition only. Total chronological supportN is5,10 or20 eligible reviews. Only firstk=1 or3 enjoyment labels are revealed; every support ID and its source-snapshot hours are available as a weak library proxy. Positives afterN are targets. All methods rank the identical full1500-item candidate universe with ALL N observed support games removed. Hidden support labels are never passed to fold-in.

Explicit labels always override implicit evidence; explicit neutral blocks positive implicit evidence and not-interested items contribute no positive feature. Weak weights are temporary inference features, never saved ratings. Uniform weighting represents observed ownership/exposure; normalized log1p(hours) divides by the largest eligible unjudged log1p value for that user.

Selected on pooled validation NDCG@20: **ease, normalized_log1p_hours, coefficient=1**. Coefficients tested:0,.05,.1,.25,.5,1; algorithms EASE/hybrid. Coefficient0 is included for both algorithms so algorithm-routing gains are not attributed to additional implicit evidence.

| Total supportN | Explicit labels k | Queries | Current explicit-only NDCG@20 | Current implicit-only NDCG@20 | Selected mixed NDCG@20 |
|---:|---:|---:|---:|---:|---:|
|5|1|100|0.0467|0.1183|0.1229|
|5|3|100|0.0845|0.1183|0.1226|
|10|1|100|0.0439|0.1127|0.1084|
|10|3|100|0.0610|0.1127|0.1011|
|20|1|100|0.0493|0.1081|0.1036|
|20|3|100|0.0585|0.1081|0.1037|

## Pooled validation

| Method | Queries | NDCG@20 | Recall@20 |
|---|---:|---:|---:|
|explicit_current_route|600|0.0573|0.0732|
|implicit_current_route|600|0.1130|0.1427|
|ease:normalized_log1p_hours:1|600|0.1104|0.1481|

## Limits

- Reviewed games are an exposure/review-selected library proxy, not a random sample of owned games
- Source snapshot hours can include playtime after the review timestamp; this is not a temporal implicit benchmark
- Chronological order is based on review dates, not acquisition or first-play dates
- Only original seed42 validation users were scored; fresh-user test must precede production activation
- Repeated users across total-support/explicit-label groups are not independent pooled observations
- Candidate universe is training-selected popular1500 games; cold/rare catalog items not evaluated

Detailed aggregate metrics/grid: `mixed-v2-validation.json`. Local selection config: `data/mixed-v2-config.json`. Keep this configuration fixed before the fresh-user test.

Reproduce: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/benchmark_mixed_v2.py --data-dir data --max-users 100`.
