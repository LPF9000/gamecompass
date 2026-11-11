# Evaluation and interpretation

Read `docs/benchmarks.md` for measured results and exact cohort limits. Metrics in test fixtures are not project performance evidence.

Population evaluation assigns entire users to training/validation/test using a seeded deterministic hash. Item eligibility/popularity is determined using training users only. A warm target is removed before population fitting. Cold-start queries reveal chronological signed support from wholly held-out users; no pretrained held-out user embedding is accessed. Per-history cohorts may contain different users: cross-row improvement is not a causal learning curve.

Candidate ranking uses all eligible items in the explicitly bounded train-only item universe, excludes revealed interactions and does not sample an artificially easy set of negatives. Noninteractions are unknown. Recall, NDCG and MAP measure recovery of observed withheld positives, not calibrated enjoyment. The negative-ranking diagnostic counts known withheld dislikes among top recommendations but cannot label unseen titles disliked.

Validation selects algorithms and hybrid settings. The test set is for reporting. Bootstrap intervals describe uncertainty within the sampled evaluation users; they do not correct selection bias, review-selection bias, exposure bias, age, missing metadata or missing-not-at-random data. Sparse and high-history cohorts are reported separately, including insufficient-data cases. Diversity is feature-based, and novelty is training-frequency-based; neither proves enjoyment.

Temporal evaluation fits only interactions before a global timestamp cutoff and predicts later positives. Timestamp-missing interactions require explicit exclusion/handling. Static metadata is not a time-travel-safe snapshot, so temporal content results carry that limitation.

The local Settings personal test hides selected known ratings without deleting them, folds in the remaining ratings and scores the hidden preferences. Separate outcome controls record perceived recommendation relevance and post-play enjoyment. Offline retrieval accuracy does not guarantee real-world satisfaction.
