# Known limitations and next experiments

This release is a working local research application. It does not establish exceptional or state-of-the-art enjoyment prediction. Results are a first reproducible baseline.

1. Default population training uses a deterministic 1/32 complete-user cohort, not all 41M source recommendations. Ranking evaluation uses 1500 training-selected games, while the local catalog contains 50,872. Bounded popular-item selection makes measured long-tail results less representative of the full catalog.
2. Review histories have selection/exposure bias; review-time playtime can include behavior after the opinion formed. Historical binary recommendation labels do not capture every dimension of enjoyment or modern console preferences.
3. High-history test cohorts are small (34 users at support50 and8 at100 in this split). Confidence intervals and different cohort composition matter. A single deterministic split is not broad external validation.
4. Collaborative models have strongest evidence only for eligible historical items. Content exploration and metadata-only support fallback handle absent items, but cold-item enjoyment accuracy is not established. The measured item-evidence ablation did not show a clear content advantage over expected random ranking; see docs/benchmarks-cold-items.md. A supervised TF-IDF-to-collaborative ridge projection improved validation and some test point estimates modestly, with broad intervals; it remains a standalone experiment (docs/content-projection.md), not an established solution. Newly cached catalog entries are represented immediately through the existing content vocabulary; population retraining is not needed for this fallback. Cold-item accuracy remains unvalidated.
5. Metadata coverage is Steam-centric. Console platform filters, alternate titles, franchise relationships, studios and artwork depend on verified enrichment. Live IGDB access requires credentials/terms acknowledgement. No fake covers or guessed identity merges.
6. Steam endpoints are implemented and mocked but not exercised with a real user's credentials. Recent/achievement views are available on demand; all achievement histories are not bulk imported. Other services do not have implemented direct authorized integrations.
7. Optional preference reasons are stored and explainable catalog overlaps are shown, but reasons have no independently validated learned effect yet. Neutral/insufficient-experience ratings are not treated as positive evidence. Not-interested is an exclusion.
8. Native closed-form fits restart after interruption; checkpointing/resumption is not implemented for every algorithm. Memory is bounded by the development cohort/item cap; full41M processing is streaming, but normalized model preparation materializes the chosen interaction cohort.
9. CPU execution was tested on Linux. CUDA, MPS and Windows/macOS end-to-end behavior remain unverified. No GPU performance claims.
10. The app is local loopback software without multiuser network authentication, encrypted database or automatic backups. No public hosting/mobile service is included.
11. Metadata/content comparisons use static snapshots, not historically versioned text. Temporal CF evaluation is available, but temporal content can leak later descriptive information.
12. Data and learned-artifact redistribution have their own licensing constraints. UCSD material remains local research-only pending clarification. The CC0 publisher declaration cannot guarantee every third-party game description/artwork right; public releases should favor downloaders and learned representations, not asset bundles.

## Experiments that can improve the evidence

- Repeat multiple user splits, bootstrap paired per-user differences and compare against stronger tuned implicit-feedback baselines.
- Expand the item universe without a dense EASE memory bottleneck; consider sparse linear recommenders and graph/approximate-neighbor retrieval.
- Tune RecBole BPR/LightGCN regularization, learning rates and epoch budgets on validation only; try learned new-user encoders rather than simple item-vector fold-in.
- Evaluate truly new users/items using time-aware metadata snapshots; distinguish release-cold items from popularity-selected existing games.
- Validate imported ownership/playtime histories against explicit feedback, model ambiguous playtime, and learn explicit/implicit confidence using a properly observed cohort.
- Acquire legitimate cross-platform explicit ratings and verify mappings for console-focused users.
- Test preference reasons and pairwise elicitation with a small voluntary local study; evaluate eventual post-play enjoyment separately from recommendation clicks.
- Fine-tune content embeddings on real preference pairs with leakage-safe splits before expecting generic sentence models to beat collaborative signals.
- Add calibrated enjoyment estimates only after validation on representative explicit labels; do not relabel raw ranking scores.
- Improve responsive background-job controls and packaging after recommendation quality is independently tested.

## V0.2 follow-up

Fresh confirmation and changed conclusions are in fresh-v2.md. Structured tags were omitted by the v1 content path; v2 corrects this and shows limited positive cold-item evidence. Mixed explicit/imported profiles are improved within their validated 5–20 mapped-game and 1–3 explicit-rating scope, with reviewed-library and snapshot-hours caveats. Broader global blending, aggregate approval and advanced neural fold-in did not justify replacing the main ranking.

The enriched ready bundle has separate Wikidata-backed console entities. Their recommendation accuracy is not measured by the Steam-only 1,500-game benchmark. Earliest entity publication, selected-platform publication and source-defined parent-work publication have distinct meanings. Original-work dates and edition kinds remain incomplete, so strict original-era/remake filters can exclude many records with unknown facts.

The main collaborative component still represents the bounded 1,500-game training universe. Ratings outside it remain stored and can inform content exploration, metadata fallback and contextual recommendations, but are not automatically assigned invented collaborative factors or forced into an unvalidated global blend. Expanding that coverage and validating partially represented profiles remain explicit follow-up work.
