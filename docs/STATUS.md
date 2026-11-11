# Acceptance status

This document distinguishes working implementation from verified accuracy and outstanding requirements. The project is not claimed complete against every ambition in the original brief.

| Acceptance requirement | Evidence / current state |
|---|---|
| Local application launches | Streamlit launcher loopback health returned HTTP200; UI workflows covered by AppTest |
| Real data acquired/processed | Kaggle v28 archive692,315,526 bytes; all41,154,794 rows scanned; deterministic complete-user1/32 cohort retained |
| Collaborative models trained | Item-kNN, EASE and SVD actual fits; RecBole BPR/LightGCN experiments documented separately |
| Baseline before local users | Versioned pretrained population artifact exists locally; clone setup acquires/trains or uses separately supplied ready data bundle |
| Independent profiles | SQLite Store + profile isolation/export/import/deletion tests |
| Search and ratings | Local50,872-game catalog; five enjoyment values plus separate activity, insufficient/not-interested |
| Personalized recommendations | Validation-selected EASE/popularity hybrid plus history routing; measured test advantage over popularity in main cohorts |
| New users without retraining | Signed item/content fold-in; whole users held out from population training |
| Updates after feedback | Inference reconstructs profile immediately; engine/UI tests |
| Played/unwanted filters | Eligibility filters and platform tests; ratings/negative exclusions |
| Offline catalog | Cached catalog and model artifacts; no live API required for inference |
| Steam integration | Authorized endpoints implemented/mocked; live key/account still needed for verification |
| Reproducible performance | Actual native warm/cold/temporal reports, provenance and exact commands |
| Multiple algorithms compared | Native six algorithms plus separate RecBole and semantic experiments |
| Selection justified | Pooled validationNDCG20 and per-history validation routing; no test-driven weight selection |
| Automated tests | See final validation record below; external API calls mocked |
| Restart persistence | SQLite profiles and portable checked NPZ/JSON model round-trip tests |
| Setup documentation | README + acquisition/research/licenses/training/evaluation/app/privacy/troubleshooting guides |
| No recurring paid service | All standard operations local; no purchases made |
| Usage restrictions respected | Source codeMIT; source rights matrix; UCSD excluded from public data/weights; credentials/data ignored |

## Important remaining verification

- Live authorized Steam and IGDB calls; user credentials and provider rights review required.
- End-to-end user satisfaction and modern console coverage, not established by Steam review retrieval.
- Full-source training, large-catalog cold-item benchmarks, multi-split external validation and stronger tuned neural baselines.
- Validated mixed explicit+implicit confidence, preference-reason learning and cross-platform imports.
- GPU/Apple Silicon and macOS/Windows verification; full resumable training for every algorithm.

Source datasets and private profiles remain outside Git. The repository contains aggregate benchmark results, source code, configuration and tests. No claim of globally best recommendation accuracy is made.

## Validation record

Main Python 3.12 suite: **58 passed, 1 skipped** (the optional real RecBole resume test requires its separate environment). RecBole Python 3.11 environment: **4 passed**, including real interrupted/resumed training equivalence. Compile checks and git whitespace checks pass. Streamlit launch health returned HTTP 200. AppTest additionally rendered Home, Discover and Settings using the real Kaggle-trained artifact. Freshly extracted ready-data bundle generated recommendations; it contains 50,872 catalog entries and no population interaction table or personal profiles.
