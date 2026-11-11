# Historical v0.1 delivery

For the current follow-up release, see followup-v2.md, fresh-v2.md and STATUS.md. Counts below record the original delivery rather than the enlarged v0.2 catalog.

Public repository: https://github.com/LPF9000/gamecompass . Execution checklist: PLAN.md. Acceptance details: docs/STATUS.md. This is a working first research release, not a claim that every long-term accuracy ambition has been achieved.

## Start

Follow the README Python 3.12 environment installation. Then either extract the separately supplied `gamecompass-ready-data.zip` in this checkout and run `gamecompass app`, or reproduce acquisition/training:

```bash
gamecompass acquire --source kaggle --user-modulus 32 --activate
gamecompass benchmark --max-items 1500 --max-users 300 --seed 42
gamecompass app
```

The ready bundle contains only catalog metadata and pretrained native model artifacts: no population interaction histories, private profiles, credentials or UCSD data. Its catalog-only database is for inference; full acquisition is required before rerunning training/evaluation.

## Implemented

Independent local profiles; ratings, reasons and independent activity; manual/Steam library imports; JSON/CSV data workflows; exclusion and platform filters; immediate explicit and implicit-only fold-in; offline catalog; metadata-only new-game fallback; grounded explanations; personal holdout testing; separate relevance and post-play outcomes; model integrity/versioning/rollback; opt-in local retraining snapshots; optional deterministic IGDB enrichment; loopback Streamlit UI and tests.

## Data and licenses

Kaggle Game Recommendations on Steam v28, publisher-declared CC0. Acquired archive: 692,315,526 bytes; scanned 41,154,794 source reviews. Retained complete histories for deterministic 1/32 users: 430,864 users, 24,868 interacted titles, 1,295,681 interactions (1,110,813 positive; 184,868 explicit negative), 50,872 catalog records. Source reviews span 2010-10-23 through 2022-12-31. Newer catalog coverage requires enrichment/manual additions.

UCSD V1 research acquisition additionally produced 58,431 interactions, 25,458 users and 32,618 catalog records. Its application/redistribution permission is unresolved; raw data and weights remain local research-only and are excluded from the ready bundle/public repo. No identity merging across source users. Exact schemas, hashes, processing statistics, download commands and rights matrix are in docs/datasets.md and docs/data-licenses.md.

MIT applies to original source code. Third-party dependencies, datasets, descriptions, artwork and models keep their own terms. Optional MiniLM weights are Apache-2.0. No purchases, subscriptions or hosted inference were used.

## Models, methodology and actual results

Population CF training: 321,213 native-matrix users and 658,217 positive interactions across 1,500 training-user-selected items. Users are hash-partitioned 80/10/10; whole cold-test users are absent from population fit. Cold support is chronological signed review history; targets are withheld positive reviews. Candidates are every eligible game with support removed, without sampled evaluation negatives. Maximum 300 test queries per history cohort; support 50 has 34 users and 100 has 8. Static metadata and review-selection/exposure bias remain limitations.

Native algorithms: popularity, item-kNN, EASE, SVD, TF-IDF and a validated hybrid. RecBole BPR/LightGCN and MiniLM run independently under the same bounded protocol where applicable. Source/split hashes, parameters, confidence intervals, warm/cold/temporal, MAP/Recall/NDCG, novelty/diversity/coverage and long-tail diagnostics are published as aggregate reports.

Pooled native validation NDCG@20: hybrid 0.0960; EASE 0.0953; item-kNN 0.0807; SVD 0.0574; popularity 0.0468; content 0.0112. Selected pooled hybrid is 90% normalized EASE plus 10% popularity, with content weight 0; per-history validation routing can choose EASE instead. This avoids replacing a measured winner with a more complex model merely for novelty.

| Revealed ratings | Production NDCG@20 | Popularity NDCG@20 |
|---|---:|---:|
| 1 | 0.1231 | 0.0643 |
| 3 | 0.0960 | 0.0502 |
| 5 | 0.1115 | 0.0502 |
| 10 | 0.1109 | 0.0398 |

These values are rankings, not enjoyment probabilities. Results use a bounded item universe and one seeded split. Confidence intervals do not establish universal superiority.

RecBole fixed 50-epoch BPR validation NDCG@20=0.03984;LightGCN=0.05298, below the native hybrid. BPR training 182.44s; LightGCN 248.50s, excluding shared preparation/evaluation. Full-batch LightGCN and BPR batch 32,768 are fixed CPU-budget experiments, not an extensively tuned neural ceiling. See docs/recbole.md.

Temporal cutoff 2022-02-16 yielded hybrid NDCG@20=0.0572 versus popularity 0.0431; CIs overlap. Implicit-only validation selects observed-game or log-hours weighting; test at one revealed observed game NDCG@20=0.1317, with explicit review-labels hidden. Reviewed histories are only a library proxy, and snapshot playtime has temporal bias. Mixed explicit+implicit confidence remains unvalidated, so explicit ratings take precedence.

Cold-item TF-IDF did not clearly beat expected random ranking. A supervised metadata-to-collaborative ridge experiment improved validation 0.0354 vs 0.0268 and some test point estimates, with broad intervals; it remains an experiment. MiniLM did not demonstrate a reliable overall gain over simpler content features. See docs/benchmarks-cold-items.md, docs/content-projection.md and docs/semantic.md.

## Verification and remaining work

Main suite: 58 passed, 1 optional RecBole-only test skipped. Separate RecBole tests: 4 passed, including exact 2-epoch versus 1+resume equivalence on an explicitly labeled fixture. Compile/whitespace checks pass. Real-model UI pages and fresh extracted runtime were exercised; Streamlit health returned HTTP 200. Remote CI status is separate and should be inspected on GitHub.

Still needed: live Steam/IGDB credentials for authorized verification; verified modern console coverage/artwork; stronger cold-item models; multi-split/external validation; larger-catalog/full-corpus training; learned preference-reason and mixed-signal effects; GPU/Apple Silicon and macOS/Windows testing. Main corpus is historical Steam, not a complete modern cross-platform catalog. No offline metric establishes eventual real-world satisfaction.

Next experiments are prioritized in docs/limitations.md. The application already works without local user growth, and personal feedback updates it immediately; exceptional recommendation quality remains an evidence-driven research goal.
