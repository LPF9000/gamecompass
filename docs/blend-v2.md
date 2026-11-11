# Validation-selected evidence combination

The existing collaborative ranking, atomic tags, descriptions, supervised projection, aggregate review approval, item neighbors and log popularity are separate channels. Weights are chosen using validation NDCG@20 across a finite, predeclared proposal grid. No fresh-test outcomes are read.

Validation queries: 450. Baseline NDCG@20: 0.099181. Best combination: 0.099194. Weights: baseline=0.900, tags=0.050, itemknn=0.050.

Default promotion requires an absolute validation gain of at least0.001, fixed before confirmation. Approved: False. Repeated validation queries are not independent evidence of generalization. Full trials: blend-v2-validation.json.

Score normalization uses the same original CF reference universe offline and during serving, before platform/seen filtering. Candidate generation unions top200 per evidence channel; reranking uses the validated combination. Fresh confirmation reports retrieval agreement separately from ranking accuracy. Approval is aggregate observed review opinion, not an enjoyment probability for an individual.
