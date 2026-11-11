# Cold-item evidence ablation

hash20% bounded-universe item holdout; cold items never occur in support and no collaborative weights, interaction counts or popularity values are used for cold scoring; warm-only signed chronological prefix; later cold positive reviews as targets; full cold-item candidate ranking; complete test users absent from population fit

No additional CF model is trained: every cold-item CF prediction is defined to have no evidence (zero). Metadata TF-IDF is unsupervised and independent of interaction labels. Existing fitted CF weights are loaded as part of the artifact but never accessed for scoring.

Zero-evidence CF and uniform prior give identical zero scores with canonical-ID tie breaking. Analytic expected random order is a mathematical expectation, not an observed experiment.

Cold candidate items: 279 of 1500 eligible items. Candidate lists contain every cold item, without sampled negatives. Support history lengths count warm eligible reviews only.

| Warm support | Queries | Content NDCG@20 (95% CI) | Content Recall@20 | Zero-evidence / uniform NDCG@20 | Expected random NDCG@20 |
|---|---:|---|---:|---:|---:|
| 1 | 300 | 0.0220 (0.0126–0.0360) | 0.0457 | 0.0207 | 0.0283 |
| 3 | 300 | 0.0349 (0.0226–0.0485) | 0.0681 | 0.0273 | 0.0301 |
| 5 | 300 | 0.0292 (0.0185–0.0407) | 0.0587 | 0.0202 | 0.0307 |

## Limits

- Conditional on users having a later cold positive; does not estimate all-user recommendation satisfaction
- Cold IDs are selected from a bounded training-positive-selected1500-item universe; not representative of truly unreleased games
- Metadata snapshot may postdate historical review timestamps; this is an item-evidence ablation, not a chronological release simulation
- No collaborative model is refit on warm items; only the cold-item no-evidence behavior is evaluated
- Hash partition and metadata parameters are fixed before test scoring; no test-tuned combination weights

This does not validate quality on actual new releases. Full metrics, confidence intervals and coverage: cold-item-benchmark-results.json.

Reproduce: `python scripts/benchmark_cold_items.py --data-dir data --max-users 300 --seed 42`.
