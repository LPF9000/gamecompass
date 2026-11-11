# Architecture and persistent state

Python services are reusable independently of Streamlit. No hosted database, external inference, microservices or automatic telemetry.

- `ingest`: authorized downloads, manifests, streaming schema normalization, statistics and RecBole atomic export.
- `catalog`: deterministic provider-ID mappings and optional credentialed IGDB enrichment; ambiguity is retained rather than title-merged.
- `store`: versioned transactional SQLite local profiles, catalog cache, explicit ratings, independent implicit signals, outcomes and exports.
- `models`: portable sparse/dense item representations with signed new-user fold-in and safe NPZ/JSON loading.
- `evaluation`: whole-user holdout, train-only item selection, warm and temporal tests and validation-only model selection.
- `engine`: per-profile inference, eligibility filters, content fallback and evidence-limited explanations.
- `integrations`: documented Steam provider, environment credentials, private-history errors.
- `personal`: local withheld-rating diagnostics and separately recorded recommendation relevance/post-play enjoyment.
- `registry`: integrity-checked model activation, activation history and optional evaluation gates.
- `ui`: local Streamlit presentation; calls services rather than embedding models or writing SQL.

`data/population.sqlite` is normalized public population research data. `data/local.sqlite` contains personal profiles and a catalog cache; local user data never joins population training automatically. Source IDs are namespaced and hashed, not merged across datasets. Steam app IDs canonicalize to `steam:<appid>`; editions/DLC keep distinct IDs.

SQLite `user_version=1` protects local schema compatibility; newer databases are rejected, not downgraded. Export JSON carries its own schema marker. A new schema revision must migrate transactionally and preserve older compatible exports.

Model artifacts under `data/models/<version>/` contain JSON and NPZ, no arbitrary pickle objects. The manifest records provenance, parameters and SHA-256 hashes. `current.json` is the active version pointer. Older versions remain available for explicit rollback. Global training runs are independent CLI jobs; the UI remains usable while they run. Downloads and imports report progress; training is not automatically triggered by a feedback edit.

Cold-start requests construct a signed item/content preference representation immediately. When explicit ratings are absent, a validation-selected implicit-only adapter can infer a weaker representation from imported observed games and playtime without storing enjoyment labels. They do not use a pretrained embedding for the local user. Ranking scores are uncalibrated and are never labeled enjoyment probabilities. Zero-history requests return a labeled population baseline. Metadata-only exploration is separate when the selected collaborative model cannot rank new items.
