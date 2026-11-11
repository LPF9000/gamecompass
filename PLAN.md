# GameCompass implementation plan

Status: working local application and real-data development benchmarks implemented. See docs/STATUS.md for verified evidence and unresolved requirements.
Owner: LPF9000. Project is free, noncommercial, local-first and privacy-conscious.
This is the execution checklist and acceptance contract. Mark an item complete only with evidence.
Unfinished or blocked requirements stay visible; no fabricated data, metrics, or integration claims.

## Product and quality contract

Build a working Python local application with an offline catalog and population-trained recommender before any local user joins. Independent profiles, manual ratings, service imports and immediate cold-start personalization are mandatory. Recommendation quality means predicting enjoyment, not maximizing playtime, popularity or clicks. Activity and enjoyment are separate signals. Unobserved games are not explicit dislikes.

No paid services, subscriptions, mandatory cloud accounts or automatic purchases. Ask before any expense, credentials, or consequential licensing decision. MIT or Apache-2.0 for original code after dependency review; third-party data, models and artwork retain their own restrictions. Never publish secrets, personal histories, restricted datasets or unapproved weights.

## 1. Environment and research

- [x] Inspect initial environment: Python 3.12.14, approximately 9.7 GiB RAM, 55 GiB free disk; no Torch or Streamlit in default Python.
- [x] Verify authenticated GitHub identity and create public repository.
- [x] Detect CPU/CUDA/MPS, record actual hardware and choose bounded smoke/development/full configurations.
- [x] Research RecBole/RecBole 2.0, PyTorch, BPR-MF, LightGCN, EASE, item-kNN, sequential/knowledge-aware methods, content models, Sentence Transformers and feature-based ranking.
- [x] Document assumptions, complexity, cold-start limitations, compatibility, licensing and primary research in docs/research.md.
- [x] Isolate RecBole if its dependencies require a different Python/NumPy runtime; do not silently replace RecBole experiments.

## 2. Legitimate data acquisition and catalog

- [x] Inspect UCSD Steam source and actual release schemas, including IDs, labels, timestamps, playtime and metadata.
- [x] Compare alternatives by preference quality, age, users/items/interactions, sparsity, size, license and reliability.
- [ ] Verify source licenses separately for local research, application use, raw redistribution and derived-weight redistribution.
- [x] Prefer explicit recommendations/ratings; preserve ownership and playtime as weaker, distinct evidence.
- [x] Implement resumable downloads, retries/backoff, SHA-256 manifests, streaming compressed parsing, schema checks, deduplication and provenance.
- [x] Acquire real data; record exact downloaded bytes, observed fields and processed statistics.
- [x] Produce versioned normalized interactions and RecBole atomic files without loading full raw datasets into RAM.
- [x] Build canonical local catalog with deterministic external-ID mappings, ambiguous-match review/confidence, editions/remakes/DLC distinctions.
- [x] Investigate IGDB authentication/terms/cache/redistribution; support authorized metadata acquisition and local inference.
- [ ] Include titles, alternate titles, dates, platforms, genres, themes, studios, modes, descriptions, franchise links, covers and gameplay characteristics when available. Missing fields stay missing.
- [x] Keep all large or restricted data outside Git. No synthetic substitution for blocked training data.

## 3. Population models and cold-start personalization

- [x] Train popularity and item-based CF baselines, EASE and BPR-MF on real data; investigate/train LightGCN using RecBole where feasible.
- [x] Keep positive, explicit-negative and unobserved signals distinct; document sampled-negative assumptions.
- [x] Implement content features and compare TF-IDF/structured features against local semantic embeddings.
- [x] Implement new-user fold-in/weighted item aggregation and signed content preference vectors without global retraining.
- [x] Handle completely unseen games using metadata, with an honest uncertainty indication.
- [ ] Build modular candidate generation and reranking.
- [x] Fit or validate blend weights on validation data only; retain a simpler model if it wins.
- [x] Persist portable, versioned artifacts with integrity checks, training provenance, atomic writes and rollback.

## 4. Evaluation before model selection

- [x] Fix seeded, reproducible train/validation/test splits before tuning.
- [x] Warm-user evaluation with sufficient histories.
- [x] Hold out entire users from population training; expose only support histories for cold-start evaluation.
- [x] Evaluate support lengths 0, 1, 3, 5, 10, 20, 50 and 100+ where data permits. Report insufficient cohorts explicitly.
- [x] Temporal evaluation with timestamp cutoffs and no future metadata/interactions leakage; disclose any static-metadata limitations.
- [x] Use identical eligible candidate universes per experiment, remove support/seen games, and document full-catalog versus sampled candidates.
- [x] Report Recall/NDCG at 10 and 20, MAP@K, coverage, diversity, novelty, history-length cohorts and niche/long-tail performance.
- [x] Include bootstrap confidence intervals where feasible and explicit-negative ranking diagnostics; numerical rating errors only for valid rating data.
- [x] Discuss exposure/selection/review bias, missing-not-at-random data, dataset age and cold items.
- [x] Save actual benchmark commands, configurations, runtimes, results and uncertainty in docs/benchmarks.md.
- [x] Select production model using measured cold-start quality and practical resource limits, without test-set tuning.
- [x] Never describe ranking scores as enjoyment probabilities unless independently calibrated.

## 5. Local service and application

- [x] Separate ingestion, catalog, profiles, models, serving, evaluation and UI behind typed internal APIs.
- [x] SQLite migrations, validated configuration, parameterized queries, structured logging, safe errors and transactional writes.
- [x] Independent local profiles with create/export/import/delete; no automatic data uploads.
- [x] Ratings: loved/liked/neutral/disliked/hated, insufficient experience, not interested; optional numeric scale and preference reasons.
- [x] Activity: owned/played/completed/abandoned/wishlist/unplayed, independent of rating.
- [x] Home: recommendations, recent ratings, profile summary and connection status.
- [x] Discover: rankings, descriptions/covers when licensed and available, platforms/themes, grounded explanations, filters/sorting.
- [x] Search: fast local title/alternate-title search and rating/status/wishlist controls.
- [x] My Games: libraries, histories, editable feedback and duplicate handling.
- [x] Settings: profiles, integrations, import/export, model/data versions, privacy and diagnostics.
- [ ] Responsive progress for long jobs; avoid blocking UI with downloads/training.
- [x] Exclude played/unwanted titles and restrict by supported platform.
- [x] Recommendations update promptly after feedback; zero-history baseline clearly labeled nonpersonalized.
- [ ] Optional informative onboarding questions without lengthy mandatory questionnaires.
- [x] Launch on loopback only; inference works offline after initial acquisition.

## 6. Integrations and personal testing

- [x] Steam provider using documented owned-games, playtime, recently played and achievement endpoints, respecting API keys and visibility.
- [x] Secure credentials through environment/keyring; never logs, Git or exports.
- [x] Review documented PlayStation/Xbox/Nintendo/GOG feasibility; no undocumented private APIs or fragile scraping.
- [x] Provide validated manual CSV/JSON imports when direct integration is unavailable.
- [x] Personal test workflow: withhold known feedback, score hidden preferences, recommend unplayed games, record relevance feedback and later enjoyment separately.
- [x] Immediate personalization plus explicit manual global retraining using versioned opt-in inputs.
- [x] Experiment tracking, checkpoints, evaluation gates and rollback; scheduled retraining optional and disabled by default.

## 7. Verification and delivery

- [x] Lock dependencies; validate backend compatibility and memory requirements.
- [x] Unit/data/model/integration/UI/end-to-end tests with mocked external services.
- [x] Cover corrupt artifacts, interrupted downloads/training, duplicates/ambiguity, missing metadata/credentials, empty/unseen users, conflicts, unsupported platforms and private histories.
- [x] Run application launch and persistence checks, real-data training and comparative benchmarks.
- [x] CI for tests/lint; no live credentials or large downloads required in ordinary tests.
- [x] README, AGENTS.md, license, gitignore, pyproject, lockfile, examples and incremental commits.
- [x] Guides: quick start, architecture, acquisition/provenance/licenses, training, evaluation, integrations, app use, privacy, troubleshooting, limitations and roadmap.
- [x] Exact install/download/process/train/benchmark/launch/import/profile/update commands.
- [x] Final report maps all definition-of-done requirements to evidence or exact outstanding blockers.

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

## Evidence checkpoint

Primary data is Kaggle CC0 v28, complete-user1/32 cohort (1,295,681 normalized reviews;430,864 users;50,872 catalog games). Native development benchmark1500eligible items selected by training users; production hybrid90%EASE+10%popularity pooled validation winner, with validation history routing. See docs/benchmarks.md and docs/STATUS.md for limits. Partial checkboxes remain open when complete requirements are not met.
