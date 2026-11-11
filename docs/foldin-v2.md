# Validation-only observed-feedback fold-in experiment

No test users were scored. This is candidate selection, not an independent generalization result. An independent evaluation must use reserved users after freezing this configuration.

Source population SHA-256: `e873ed8dc766549b66d69496a6580d38242fa84786f6ce029609241b94e0a76f`.
Candidate universe: 1,500; pooled support lengths 1/3/5 validation queries: 900; max 300 per history cohort.

Only supplied observed positive/negative support labels fit user factors. Unobserved candidates never become negative examples.

Entire existing training-user-selected CF item universe; support items removed identically for every model. No sampled evaluation negatives.

| Baseline | Validation NDCG@20 | Recall@20 |
|---|---:|---:|
| popularity | 0.046777 | 0.080866 |
| ease | 0.095259 | 0.152587 |
| hybrid | 0.096014 | 0.156383 |
| production_routed | 0.096129 | 0.155587 |

## Best fold-in candidates

| Factor model | User method | Item normalization | Regularization | Validation NDCG@20 | Recall@20 |
|---|---|---|---:|---:|---:|
| LightGCN | logistic | none | 10 | 0.055180 | 0.098057 |
| LightGCN | logistic | none | 0.1 | 0.054770 | 0.094433 |
| LightGCN | logistic | none | 0.01 | 0.054752 | 0.093407 |
| LightGCN | ridge | none | 1 | 0.054646 | 0.094696 |
| LightGCN | logistic | none | 1 | 0.054596 | 0.096910 |
| LightGCN | ridge | none | 10 | 0.054228 | 0.096912 |
| LightGCN | logistic | none | 0.001 | 0.054154 | 0.092013 |
| LightGCN | logistic | none | 0.0001 | 0.053727 | 0.089810 |
| LightGCN | ridge | none | 100 | 0.053407 | 0.095420 |
| LightGCN | logistic | none | 100 | 0.053112 | 0.095075 |
| LightGCN | sum | none | 1 | 0.052985 | 0.093993 |
| LightGCN | ridge | none | 0.1 | 0.052945 | 0.091179 |

## Frozen configuration

`{"config": {"max_iterations": 100, "method": "logistic", "negative_weight": 1.0, "normalization": "none", "regularization": 10.0}, "source_model": "LightGCN"}`

Frozen JSON SHA-256: `5728097aa1eeab4edb90a5297dc88e0222e43d2cfd97c3ec2d64c531c5fa5bd8`. Local artifact: `data/foldin-v2/frozen-config.json`; full grid/results: `data/foldin-v2/validation-results.json`.

Methods: original signed sum; separately averaged positive and negative centroids; ridge regression of signed observed labels with magnitude confidence; convex regularized logistic regression of those same observed labels. Transformations are raw item factors, item L2 normalization, or training-item mean centering followed by L2 normalization. Regularization grid: 0.0001/0.001/0.01/0.1/1/10/100. Logistic optimization uses at most 100 L-BFGS iterations; convergence counts are recorded per trial. No unknown interaction is fitted as a negative. All methods accept only item IDs/factors plus the supplied support, never pretrained user IDs/vectors.

The grid is selected on pooled validation NDCG@20 only. Bootstrap intervals remain descriptive because users can repeat across support cohorts and model selection is performed on this same validation set. The old inspected test results are not used in this experiment.

Reproduce: `OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/benchmark_foldin_v2.py --max-items 1500 --max-users 300 --seed 42`.


Best observed-only fold-in validation NDCG@20: 0.055180; strongest existing baseline (production_routed): 0.096129. The tested fold-in alternatives do not justify replacing the production baseline. This is a validation decision, not a fresh-test result or a claim about every neural model.

- Validation model selection is optimistic; independent reserved test evaluation is required after freezing configuration.
- Neither item-factor transformations nor new-user fits access population user embeddings.
- Logistic scores are uncalibrated user-specific logits, not enjoyment probabilities.
- Historical review selection, deterministic source-user subsampling, metadata timing and bounded candidate scope remain limitations.
