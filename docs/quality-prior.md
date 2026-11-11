# Aggregate review approval prior

An aggregate approval channel is implemented separately from review count and collaborative personalization. On the original validation users, the best tested blend improves pooled NDCG@20 by only **0.0000861 absolute**; its descriptive paired query-bootstrap interval crosses zero. This does **not** establish a meaningful accuracy gain. The existing production model is not changed by this experiment.

## API and artifact

```python
from pathlib import Path
from gamecompass.quality import QualityPrior
prior = QualityPrior.fit(prepared['train_rows'], prepared['cf_ids'])
approval = prior.score(['steam:10', 'unknown'], strength=20)
channels = prior.channels(['steam:10', 'unknown'], strength=20)
prior.save(Path('data/quality/prior'))
restored = QualityPrior.load(Path('data/quality/prior'))
```

`score` returns the Beta-shrunk rate of positive **observed source reviews**. `channels` exposes `approval`, `centered_approval`, `centered_log_odds`, `count`, `logcount`, `positive_count` and `negative_count`. Unknown or unobserved games receive the global approval prior, zero centered score and **zero evidence count**. Their lack of evidence is never treated as approval or dislike.

The artifact uses JSON game IDs/global statistics and checksummed numeric NPZ counts, with `allow_pickle=False`. It contains no source user identifiers, local profiles or raw histories. Corruption, invalid array shapes/counts, unsupported formats and invalid prior strengths are rejected. Four targeted tests verify counts, shrinkage, unknown items, scope exclusion and safe persistence.

## Estimator and semantics

For item positive count p, negative count n, global training approval g and configurable prior strength s:

`approval = (p + s*g) / (p+n+s)`

This is a Beta prior with alpha=s*g and beta=s*(1-g), followed by the item observations. Global approval is clipped to [1e-6, 1-1e-6] only for degenerate all-positive/all-negative input, preventing improper priors and infinite logits. No clipping was needed in the actual dataset. Empty training data uses g=0.5. Counts and log1p(count) remain independent channels so a high-volume game with mixed reviews can be distinguished from a small game with strong observed approval.

Observed approval is an opinion aggregate affected by who purchased, played and chose to review a game, exposure, era and source selection. It is **not a calibrated probability of individual enjoyment**, and is not an objective universal quality score. Review volume is at most an observed notability/exposure signal; it does not establish enjoyment. Personal recommendation should combine such channels with evidence about an individual’s tastes, rather than allowing either global count or approval to dominate.

## Reproducible validation-only experiment

```bash
python -m gamecompass.quality --data-dir data --report-dir docs --max-users 300
```

Source: Kaggle Steam v28, deterministic complete-user1/32 cohort. Frozen baseline artifact `development-1500-42-20261111T004254213166Z`, selected algorithm **hybrid**. Seed42; **900 validation queries**, up to300 each with chronological explicit support lengths1,3,5. Canonical candidate universe:1,500 training-selected games; support games excluded; deterministic canonical-ID tie breaks.

Only `prepare(...)[train_rows]`, after whole-user partitioning and training warm holdouts, contributes approval counts. Validation and test labels never contribute counts. Only original validation users are scored in this experiment; **no test scoring occurred**. The unchanged original validation partition is reused for model selection, so these results are tuning evidence, not fresh generalization evidence.

Grid: prior strengths5/20/100, quality blend weights0/.05/.1/.25/.5. Standalone approval (weight1) is an additional diagnostic, not eligible for the blend selection. The quality channel is logit(approval)-logit(global approval), divided by its maximum absolute value over eligible CF games. Baseline scores are independently max-absolute scaled over the same eligible candidates for each query. Blend=(1-weight)*baseline+weight*quality. Weight0 preserves the baseline ranking.

| Prior strength | Quality weight | NDCG@20 | Recall@20 | Explicit-negative fraction@20 |
|---:|---:|---:|---:|---:|
| 5 | 0 | 0.096014 | 0.156383 | 0.002111 |
| 5 | 0.05 | 0.095848 | 0.155086 | 0.001778 |
| 5 | 0.1 | 0.094733 | 0.151727 | 0.001556 |
| 5 | 0.25 | 0.088925 | 0.137413 | 0.001167 |
| 5 | 0.5 | 0.056475 | 0.074570 | 0.000389 |
| 5 | 1 | 0.005088 | 0.008776 | 0.000000 |
| 20 | 0 | 0.096014 | 0.156383 | 0.002111 |
| 20 | 0.05 | 0.096100 | 0.156197 | 0.001778 |
| 20 | 0.1 | 0.094741 | 0.151828 | 0.001556 |
| 20 | 0.25 | 0.091143 | 0.141840 | 0.001278 |
| 20 | 0.5 | 0.069035 | 0.100943 | 0.000556 |
| 20 | 1 | 0.016584 | 0.023950 | 0.000333 |
| 100 | 0 | 0.096014 | 0.156383 | 0.002111 |
| 100 | 0.05 | 0.096082 | 0.156223 | 0.001778 |
| 100 | 0.1 | 0.095131 | 0.153245 | 0.001556 |
| 100 | 0.25 | 0.091970 | 0.144847 | 0.001333 |
| 100 | 0.5 | 0.073922 | 0.112220 | 0.000722 |
| 100 | 1 | 0.026920 | 0.044047 | 0.000278 |

Grid choice: prior strength**20**, quality weight**0.05**. Baseline NDCG@20**0.096014**; selected blend**0.096100**. Paired change**0.0000861**; 1,000-resample query-bootstrap95% interval**[-0.0015366, 0.0017503]**. The selected blend slightly lowers Recall@20. Larger quality weights substantially hurt ranking retrieval, despite reducing the fraction of explicit-negative targets recommended.

The interval is descriptive: a user can occur at several support lengths, and hyperparameters were selected on these same queries. It is not a clustered independent-user significance test and does not adjust for selection across the grid. No production improvement claim follows. A preregistered fresh-user evaluation of a frozen blend or learned reranker is needed.

## Aggregate training evidence

Counts include **658,217 positive** and **109,123 negative** training labels across1,500 items. Global observed approval is**0.857791**. No user IDs are saved in the report.

| Quantile | Review count per item | Posterior approval, strength20 |
|---:|---:|---:|
| 0 | 80.0 | 0.135112 |
| 0.25 | 137.0 | 0.817464 |
| 0.5 | 226.0 | 0.889523 |
| 0.75 | 548.5 | 0.935775 |
| 0.9 | 1194.4 | 0.957906 |
| 1 | 7647.0 | 0.992251 |

Elapsed time: **25.108 seconds** in this environment, including source preparation and validation. Machine-readable complete aggregates: [quality-prior-results.json](quality-prior-results.json). Locally saved aggregate prior:`data/quality/prior/`; validation-selected config:`data/quality/selected.json`.

Full-catalog approval for non-CF items, confidence-aware quality reranking, review-era/selection effects and independent-user uncertainty remain future work. Missing-game fallback is implemented but cannot supply item-specific quality evidence without legitimate training observations.
