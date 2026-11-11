# Fresh-user v2 confirmation

All compared configurations were frozen before these users were scored. The population split, historical source and 1,500-item candidate universe are unchanged. We excluded 1,351 users whose prior results had been inspected and sampled remaining users by a prospective identity hash. This is fresh confirmation within the same dataset, not an independent dataset.

## Global ranking control

| Explicit history | Users | Existing routed model NDCG@20 | Proposed blend | Popularity |
|---|---:|---:|---:|---:|
| 0 | 500 | 0.0613 | 0.0613 | 0.0613 |
| 1 | 500 | 0.0792 | 0.0783 | 0.0531 |
| 3 | 500 | 0.1131 | 0.1117 | 0.0451 |
| 5 | 500 | 0.0987 | 0.0978 | 0.0404 |
| 10 | 500 | 0.0982 | 0.0973 | 0.0422 |
| 20 | 51 | 0.1007 | 0.1013 | 0.0351 |
| 50 | 0 | unavailable | unavailable | unavailable |
| 100 | 0 | unavailable | unavailable | unavailable |

The proposed global blend failed its predeclared minimum validation gain and was not promoted. Fresh results also do not establish a general advantage. More complex neural fold-in remains below the control. The quality prior remains a separate diagnostic. Candidate retrieval retained 100% of the proposed blend’s top20 across these bounded confirmation cohorts. No fresh results were used to change weights.

## Metadata-only items

Every cold item was excluded from content-projection collaborative fitting. Candidates here are the same 279 deliberately withheld historically observed games, not actual future releases. Each query must have a later cold positive.

| Warm history | Users | Previous description baseline | Structured projection | Paired NDCG@20 change (95% interval) |
|---|---:|---:|---:|---|
| 1 | 173 | 0.0254 | 0.0522 | +0.0268 (+0.0071, +0.0467) |
| 3 | 204 | 0.0236 | 0.0448 | +0.0212 (+0.0033, +0.0382) |
| 5 | 182 | 0.0321 | 0.0394 | +0.0073 (-0.0118, +0.0285) |

The one- and three-history paired intervals exclude zero in this fixed experiment; the five-history interval does not. This is limited evidence, not a claim of universally strong cold-item accuracy. Static metadata, selected targets and multiple related comparisons remain limitations. The selected projection was chosen on validation; the test did not choose a different method for the five-history cohort. It is used for content exploration and metadata fallback, while the main CF ranking stays unchanged.

## Explicit ratings plus imported history

All methods exclude the same full support history from candidates. Only the first k review labels are revealed; other support labels are hidden while game IDs and snapshot hours serve as an imported-library proxy.

| Total known games | Explicit ratings | Users | Explicit-only NDCG@20 | Mixed | Implicit-only diagnostic |
|---|---:|---:|---:|---:|
| 5 | 1 | 500 | 0.0598 | 0.1118 | 0.1102 |
| 5 | 3 | 500 | 0.0748 | 0.1005 | 0.1102 |
| 10 | 1 | 500 | 0.0504 | 0.1041 | 0.1091 |
| 10 | 3 | 500 | 0.0628 | 0.1010 | 0.1091 |
| 20 | 1 | 51 | 0.0617 | 0.1138 | 0.1222 |
| 20 | 3 | 51 | 0.0404 | 0.1075 | 0.1222 |

Mixed evidence improves over the previous explicit-only behavior in these groups, with positive per-group paired intervals. It does not consistently beat the implicit-only diagnostic. Explicit judgments must still override weak observations on the same game; neutral and not-interested records cannot become inferred likes. The deployed mixed adapter is bounded to 5–20 CF-mapped evidence items and 1–3 nonneutral explicit ratings. Other profiles retain the earlier validated behavior. No ownership/playtime weights are written as enjoyment labels.

## Limits and reproducibility

- All feedback comes from historical, exposure-selected Steam reviews. Snapshot hours may include activity after a review date.
- Cohorts overlap, so differences between history lengths are not a causal learning curve. Per-cohort bootstrap intervals use distinct users; they do not establish general statistical certainty across all comparisons.
- No unexposed 50/100-history cohorts remain; those cells are unavailable.
- Console metadata was acquired after these tests and has no measured console-specific enjoyment accuracy.
- Full aggregate Recall/NDCG/MAP, known-negative diagnostics and paired intervals are in fresh-v2-results.json.

Exact frozen configuration and protocol hashes are recorded in the aggregate report. Reproduction instructions are in followup-v2.md.
