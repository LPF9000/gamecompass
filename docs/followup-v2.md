# Follow-up release v0.2

This release adds structured gameplay tags, a validation-selected metadata-to-preference projection, bounded mixed explicit/implicit personalization, and simple candidate refinements. It preserves the previous primary collaborative model because the proposed global blend did not earn promotion.

The old feature path collected `metadata.tags` but did not include them in the content vectorizer. V2 has separate fitted blocks for 441 atomic tags and 12,000 description terms/bigrams. Coefficients, vocabularies, sparse catalog representations and checksums are portable JSON/NPZ, without Torch or remote inference in the application.

Read fresh-v2.md for prospective confirmation and caveats. Quality and neural-fold-in experiments are documented even where they did not improve the control. Choices were frozen before fresh-user predictions; test scores did not tune their parameters.

## Application

Discover offers Platform, Genre and Era, with optional Edition details. Search and ratings remain catalog-wide. Original-work, edition/entity and qualified platform-release dates remain distinct. Unknown dates cannot satisfy the corresponding era criterion. See refinements.md.

“If you liked this game” uses the selected title as a temporary seed with the same candidate refinements and saved exclusions. It never writes a rating. Metadata fallback is automatic if no collaborative candidates satisfy requested refinements. Related source records are conservatively excluded with previously played games; turning that exclusion off allows exploring other releases. Enjoyment labels and latent scores are never transferred across uncertain editions.

Wikidata provides separate source-entity entries with their own console facts; they are not silently merged into Steam editions. Acquisition uses official CC0 structured data and retains exact-ID relationships, qualifiers, statement IDs, precision and source references. Original-work dates remain unknown unless supported by positive evidence. A global entity date never becomes a console port date: a platform-specific era requires an explicitly qualified date. See wikidata.md.

## Run the pretrained application

Update the source checkout and use the latest gamecompass-ready-data.zip. Extract it into the checkout so data/models/current.json points to the v2 runtime; install the README dependencies and launch:

```bash
gamecompass app
```

The bundle contains catalog-only SQLite and native/content artifacts, with no population histories, profiles, credentials or private test-user manifests. To retrain, reacquire the population data first. Where a prior v1 model directory is present, it remains available for rollback with `gamecompass activate-model VERSION`.

## Reproduce research from a source checkout

Use the original Kaggle seed42 population fit, after acquisition and the native benchmark. All experiments explicitly separate training users, validation selection and reserved confirmation.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/train_content_v2.py --data-dir data --max-users 300 --catalog-users 100 --seed 42
python -m gamecompass.quality --data-dir data --report-dir docs --max-users 300
python scripts/benchmark_mixed_v2.py --data-dir data --max-users 100 --seed 42
```

The first command prints the new content artifact path. Substitute that path below. Preserve the original base artifact and split. Create the private protocol with `gamecompass.protocol.create_protocol(data_dir, base_version=ORIGINAL_BASE_VERSION)`. It publishes immutable user cohorts beneath ignored data/protocols/ and an aggregate summary only. Existing protocol/config files reject conflicting replacements.

```bash
python scripts/train_blend_v2.py --data-dir data --content-path CONTENT_ARTIFACT_PATH --max-users 150
```

Optional neural comparison needs the separately documented RecBole environment and original trained BPR/LightGCN factors:

```bash
python scripts/benchmark_foldin_v2.py --data-dir data --max-items 1500 --max-users 300 --seed 42
python scripts/confirm_v2.py --data-dir data --content-path CONTENT_ARTIFACT_PATH
python scripts/activate_v2.py --data-dir data --content-path CONTENT_ARTIFACT_PATH
```

`confirm_v2.py` freezes content, blend, quality, neural and mixed configurations before scoring. It is a reproducer of this frozen experiment, not a mechanism to repeatedly tune against the same reserved users. Future research must record newly exposed users and define a new untouched cohort.

Optional console metadata enrichment uses `scripts/enrich_wikidata.py`; see wikidata.md for exact arguments. It writes only public catalog games, preserving training interactions and personal profiles. `scripts/package_runtime.py --catalog-source data/local.sqlite` packages catalog rows and runtime artifacts, never profile tables.
