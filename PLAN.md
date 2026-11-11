# GameCompass implementation plan

Status: implementation has not begun. Repository created; initial environment inspected.
Owner: LPF9000. Project is free, noncommercial, local-first and privacy-conscious.
This is the execution checklist and acceptance contract. Mark an item complete only with evidence.
Unfinished or blocked requirements stay visible; no fabricated data, metrics, or integration claims.

## Product and quality contract

Build a working Python local application with an offline catalog and population-trained recommender before any local user joins. Independent profiles, manual ratings, service imports and immediate cold-start personalization are mandatory. Recommendation quality means predicting enjoyment, not maximizing playtime, popularity or clicks. Activity and enjoyment are separate signals. Unobserved games are not explicit dislikes.

No paid services, subscriptions, mandatory cloud accounts or automatic purchases. Ask before any expense, credentials, or consequential licensing decision. MIT or Apache-2.0 for original code after dependency review; third-party data, models and artwork retain their own restrictions. Never publish secrets, personal histories, restricted datasets or unapproved weights.

## 1. Environment and research

- [x] Inspect initial environment: Python 3.12.14, approximately 9.7 GiB RAM, 55 GiB free disk; no Torch or Streamlit in default Python.
- [x] Verify authenticated GitHub identity and create public repository.
- [ ] Detect CPU/CUDA/MPS, record actual hardware and choose bounded smoke/development/full configurations.
- [ ] Research RecBole/RecBole 2.0, PyTorch, BPR-MF, LightGCN, EASE, item-kNN, sequential/knowledge-aware methods, content models, Sentence Transformers and feature-based ranking.
- [ ] Document assumptions, complexity, cold-start limitations, compatibility, licensing and primary research in docs/research.md.
- [ ] Isolate RecBole if its dependencies require a different Python/NumPy runtime; do not silently replace RecBole experiments.

## 2. Legitimate data acquisition and catalog

- [ ] Inspect UCSD Steam source and actual release schemas, including IDs, labels, timestamps, playtime and metadata.
- [ ] Compare alternatives by preference quality, age, users/items/interactions, sparsity, size, license and reliability.
- [ ] Verify source licenses separately for local research, application use, raw redistribution and derived-weight redistribution.
- [ ] Prefer explicit recommendations/ratings; preserve ownership and playtime as weaker, distinct evidence.
- [ ] Implement resumable downloads, retries/backoff, SHA-256 manifests, streaming compressed parsing, schema checks, deduplication and provenance.
- [ ] Acquire real data; record exact downloaded bytes, observed fields and processed statistics.
- [ ] Produce versioned normalized interactions and RecBole atomic files without loading full raw datasets into RAM.
- [ ] Build canonical local catalog with deterministic external-ID mappings, ambiguous-match review/confidence, editions/remakes/DLC distinctions.
- [ ] Investigate IGDB authentication/terms/cache/redistribution; support authorized metadata acquisition and local inference.
- [ ] Include titles, alternate titles, dates, platforms, genres, themes, studios, modes, descriptions, franchise links, covers and gameplay characteristics when available. Missing fields stay missing.
- [ ] Keep all large or restricted data outside Git. No synthetic substitution for blocked training data.

## 3. Population models and cold-start personalization

- [ ] Train popularity and item-based CF baselines, EASE and BPR-MF on real data; investigate/train LightGCN using RecBole where feasible.
- [ ] Keep positive, explicit-negative and unobserved signals distinct; document sampled-negative assumptions.
- [ ] Implement content features and compare TF-IDF/structured features against local semantic embeddings.
- [ ] Implement new-user fold-in/weighted item aggregation and signed content preference vectors without global retraining.
- [ ] Handle completely unseen games using metadata, with an honest uncertainty indication.
- [ ] Build modular candidate generation and reranking.
- [ ] Fit or validate blend weights on validation data only; retain a simpler model if it wins.
- [ ] Persist portable, versioned artifacts with integrity checks, training provenance, atomic writes and rollback.

## 4. Evaluation before model selection

- [ ] Fix seeded, reproducible train/validation/test splits before tuning.
- [ ] Warm-user evaluation with sufficient histories.
- [ ] Hold out entire users from population training; expose only support histories for cold-start evaluation.
- [ ] Evaluate support lengths 0, 1, 3, 5, 10, 20, 50 and 100+ where data permits. Report insufficient cohorts explicitly.
- [ ] Temporal evaluation with timestamp cutoffs and no future metadata/interactions leakage; disclose any static-metadata limitations.
- [ ] Use identical eligible candidate universes per experiment, remove support/seen games, and document full-catalog versus sampled candidates.
- [ ] Report Recall/NDCG at 10 and 20, MAP@K, coverage, diversity, novelty, history-length cohorts and niche/long-tail performance.
- [ ] Include bootstrap confidence intervals where feasible and explicit-negative ranking diagnostics; numerical rating errors only for valid rating data.
- [ ] Discuss exposure/selection/review bias, missing-not-at-random data, dataset age and cold items.
- [ ] Save actual benchmark commands, configurations, runtimes, results and uncertainty in docs/benchmarks.md.
- [ ] Select production model using measured cold-start quality and practical resource limits, without test-set tuning.
- [ ] Never describe ranking scores as enjoyment probabilities unless independently calibrated.

## 5. Local service and application

- [ ] Separate ingestion, catalog, profiles, models, serving, evaluation and UI behind typed internal APIs.
- [ ] SQLite migrations, validated configuration, parameterized queries, structured logging, safe errors and transactional writes.
- [ ] Independent local profiles with create/export/import/delete; no automatic data uploads.
- [ ] Ratings: loved/liked/neutral/disliked/hated, insufficient experience, not interested; optional numeric scale and preference reasons.
- [ ] Activity: owned/played/completed/abandoned/wishlist/unplayed, independent of rating.
- [ ] Home: recommendations, recent ratings, profile summary and connection status.
- [ ] Discover: rankings, descriptions/covers when licensed and available, platforms/themes, grounded explanations, filters/sorting.
- [ ] Search: fast local title/alternate-title search and rating/status/wishlist controls.
- [ ] My Games: libraries, histories, editable feedback and duplicate handling.
- [ ] Settings: profiles, integrations, import/export, model/data versions, privacy and diagnostics.
- [ ] Responsive progress for long jobs; avoid blocking UI with downloads/training.
- [ ] Exclude played/unwanted titles and restrict by supported platform.
- [ ] Recommendations update promptly after feedback; zero-history baseline clearly labeled nonpersonalized.
- [ ] Optional informative onboarding questions without lengthy mandatory questionnaires.
- [ ] Launch on loopback only; inference works offline after initial acquisition.

## 6. Integrations and personal testing

- [ ] Steam provider using documented owned-games, playtime, recently played and achievement endpoints, respecting API keys and visibility.
- [ ] Secure credentials through environment/keyring; never logs, Git or exports.
- [ ] Review documented PlayStation/Xbox/Nintendo/GOG feasibility; no undocumented private APIs or fragile scraping.
- [ ] Provide validated manual CSV/JSON imports when direct integration is unavailable.
- [ ] Personal test workflow: withhold known feedback, score hidden preferences, recommend unplayed games, record relevance feedback and later enjoyment separately.
- [ ] Immediate personalization plus explicit manual global retraining using versioned opt-in inputs.
- [ ] Experiment tracking, checkpoints, evaluation gates and rollback; scheduled retraining optional and disabled by default.

## 7. Verification and delivery

- [ ] Lock dependencies; validate backend compatibility and memory requirements.
- [ ] Unit/data/model/integration/UI/end-to-end tests with mocked external services.
- [ ] Cover corrupt artifacts, interrupted downloads/training, duplicates/ambiguity, missing metadata/credentials, empty/unseen users, conflicts, unsupported platforms and private histories.
- [ ] Run application launch and persistence checks, real-data training and comparative benchmarks.
- [ ] CI for tests/lint; no live credentials or large downloads required in ordinary tests.
- [ ] README, AGENTS.md, license, gitignore, pyproject, lockfile, examples and incremental commits.
- [ ] Guides: quick start, architecture, acquisition/provenance/licenses, training, evaluation, integrations, app use, privacy, troubleshooting, limitations and roadmap.
- [ ] Exact install/download/process/train/benchmark/launch/import/profile/update commands.
- [ ] Final report maps all definition-of-done requirements to evidence or exact outstanding blockers.

## Workflow and change tracking

Follow stages in order when dependencies require it; independent research/implementation may proceed in parallel using agreed schemas. Update this file and docs/STATUS.md after meaningful milestones. Every reported benchmark must name its real dataset, scope, split, candidate strategy and artifact version. Smoke/development subsets must never be presented as full-dataset results.

If an external dependency blocks progress, preserve completed work, implement resumable handling, record the precise failing action and required user input, and continue independent authorized work. Do not mark the overall project complete until every essential acceptance criterion is verified.

## Initial known risks (unresolved)

- UCSD public access and citation instructions alone do not establish permission to redistribute datasets or trained weights.
- RecBole dependency constraints may require a dedicated Python environment.
- Gaming-service authentication and visibility settings can limit imports.
- Historical Steam coverage alone does not provide current console coverage.
- Approximately 10 GiB RAM constrains dense EASE item matrices and full-scale neural experiments.
- No recommendation accuracy claims are justified before real measurements.
