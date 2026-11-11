# Fresh-user confirmation protocol

The initial results are development experiments. Inspecting those results while deciding the next set of algorithms creates an adaptive research-selection effect even when each individual script tuned hyperparameters on validation only. Those original test users are now treated as previously exposed research data, not as a fresh confirmation sample. Previously reported metrics remain real measurements; they must not be relabeled as untouched confirmation evidence.

No models have been scored on the cohorts described below by the protocol-creation code. `gamecompass.protocol` only reconstructs prior exposure, samples unused users and locks provenance. Final comparisons must freeze every configuration using validation users before accessing fresh test scores. Do not select a new production winner by picking the largest fresh-test number.

## Frozen population and exposure exclusions

The original population fit, eligible item universe and seed-42 training/validation/test assignment remain unchanged. The base artifact is `development-1500-42-20261111T004254213166Z`. Repartitioning users with another seed would let previously trained users enter a purported cold-user test; this protocol deliberately avoids that error.

Exposure reconstruction unions whole users from:

- The first 300 eligible sorted primary-dataset users at every historical support length: 0, 1, 3, 5, 10, 20, 50 and 100. This includes native, implicit-library-proxy, RecBole and semantic comparisons sharing these eligibility rules. A user inspected under any one support length is excluded at every new support length.
- The first 300 eligible sorted cold-item users for warm support lengths 1, 3 and 5. Their eligibility differs from ordinary chronological-prefix queries and must be reconstructed separately.
- Previously evaluated primary temporal users that overlap the original whole-user test partition, reconstructed from the published cutoff and candidate universe. This conservatively accounts for additional already-observed information. UCSD reports use a different dataset source hash and are not conflated with the primary Kaggle users.

The exposure union contains 1,351 original test users. The temporal comparison exposed 32 original test users before unioning (some overlap other exclusions). The original split has 40,140 test users; 38,789 remain unexposed. Early UCSD experiments' cap of 500 does not imply a primary Kaggle exposure cap of 500. The primary cap was 300.

## Prospective sample

Eligible users have more than h retained eligible-item reviews and at least one positive review after their chronological h-review prefix. Histories retain signed recommendation labels; activity is not a label. The sample order is ascending SHA256 of `fresh-user-v2:20261110:<user_id>`, followed by user ID as an impossible-in-practice hash-collision tie break. Selection is independent of model predictions. Cap each history cohort at 500. Include all eligible users if fewer exist. Never repeat or synthesize users to populate an unavailable history bucket.

| Revealed eligible reviews | Remaining eligible users | Frozen final-test users |
|---:|---:|---:|
| 0 | 35,617 | 500 |
| 1 | 14,971 | 500 |
| 3 | 5,465 | 500 |
| 5 | 2,650 | 500 |
| 10 | 697 | 500 |
| 20 | 51 | 51 |
| 50 | 0 | 0 |
| 100 | 0 | 0 |

Seed: **20261110**. Protocol SHA256: `7fbc64536849ba16d71cf8e2315034c32488ab466b6801e6d8134614a1c2e4d6`.

Dataset SHA256: `e873ed8dc766549b66d69496a6580d38242fa84786f6ce029609241b94e0a76f`.

Original split SHA256: `d73cdf559e70d7eb1c3b2aa7cf43b3985b4cb77f1b031a3a5d04a41d43db0ae0`.

User IDs are stored only in ignored local `data/protocols/fresh-users-v2.json`. Public documentation contains aggregate counts and hashes. The manifest includes model weight hashes and the exact item-universe hash. It is created exclusively and cannot be replaced through the protocol API with different contents. A matching rerun is idempotent; changed data, split or weights fail verification.

## Execution contract

```python
import json
from pathlib import Path
from gamecompass.protocol import (
    create_protocol, cohort_users, freeze_experiment_config, public_summary,
)
from gamecompass.evaluation import cold_queries

data = Path("data")
manifest = create_protocol(data)  # validates existing immutable contents
split = json.loads((data / "experiments" / manifest["base_version"] / "split.json").read_text())

# Do all hyperparameter selection and feature experiments on validation users.
# config must include every tested parameter and fitted artifact checksum.
freeze_experiment_config(data, "example-comparison", {
    "algorithm": "ease", "regularization": 100.0,
    "selection": "locked using validation only",
}, manifest)

fresh = cohort_users(manifest, split, 3)
queries = cold_queries(fresh, 3, max_users=manifest["cap_per_history"])
assert len(queries) == len(fresh)
# Only after *all* compared configurations have been frozen: score queries.
```

`cohort_users()` filters the original split to exactly the locked identities and checks exposure and eligibility. Existing query builders sort IDs internally; pass a cap at least as large as the frozen cohort, so that sorting cannot choose another truncated subset. The hash-selected membership remains identical. Do not run a full `run_benchmark()` as a replacement: that would refit and score the previously exposed sorted test sample.

Compare candidates using every original eligible CF item, removing revealed support items, with canonical-ID deterministic tie resolution. No sampled negatives are used. Report Recall@10/20, NDCG@10/20, MAP@10/20, explicit-negative fraction, coverage, novelty, diversity and long-tail performance as appropriate. Unobserved items remain unknown distractors, not proved dislikes. Use paired per-user model differences for comparison confidence intervals. If pooling several history lengths, bootstrap unique users jointly across their queries; treating repeated users as independent understates uncertainty. History-20 results have only 51 untouched users and must report that limitation. History-50/100 confirmation results are unavailable.

An experimental winner should be selected on validation alone. Fresh tests estimate the effect of that locked selection and of prespecified alternative comparisons. If the test motivates another modeling choice, record it as exploratory and obtain another untouched cohort before claiming confirmation. This new sample is from the same restricted historical dataset; it is not external validation or a guarantee of enjoyment after playing an unplayed recommendation.

## Audit of existing evaluation and features

| Finding | Interpretation / required handling |
|---|---|
| Initial primary cohorts use sorted anonymized user IDs | Deterministic but potentially correlated with source collection order; new hash ordering removes this selection rule |
| Whole test users are absent from population training and positive counts | Valid cold-user separation; preserve original split and frozen weights |
| Item universe is the top 1,500 training-positive games | Ranking has full candidates within this scope, not across all catalog games; low-count and genuinely cold items are omitted |
| Support lengths count reviews inside that universe | A nominal 5-review user may have more than five real-world games; report this conditioning |
| Users need a remaining positive target | Necessary for positive retrieval metrics but selects positively reviewed users; report eligibility counts |
| Unobserved items are full-catalog distractors | Avoids artificially easy sampled-negative ranking; does not resolve exposure bias or missing-not-at-random labels |
| Implicit support uses reviewed IDs and snapshot hours | A library proxy, not actual ownership history; hours may include activity after the review date |
| Static metadata can postdate interactions | Content comparisons are historical-preference retrieval, not a strict as-of temporal simulation |
| Hybrid weights are tuned on pooled 1/3/5 validation queries | Repeated users across histories are not independent; validation confidence intervals require grouped resampling |
| `game_text()` reads descriptions, genres and themes, while Kaggle tags reside in `metadata.tags` and structured genres/themes are empty | Existing primary content experiments effectively ignore tags. A tag-aware experiment is warranted, but must use new fitted feature artifacts and validation-only selection; do not silently alter the old model |
| Native validation ranks with a CF-only model view, while serving can include a full catalog | Min/max or max-absolute feature normalization can depend on items outside the ranking universe. Check that a prospective hybrid normalizes against a fixed documented universe, so offline and online weights have the same meaning |
| Cold-item experiment scores metadata without fitting any cold CF evidence | An item-evidence ablation, not a model retrained with a real release-date holdout; cold IDs were selected from a training-positive-selected universe |
| Temporal comparison inherits earlier validation hyperparameters | No temporal-target tuning inside that script, but inherited tuning may use later-era validation evidence; do not present it as fully prospective deployment simulation |

The new protocol does not retroactively fix these development limitations. It creates a guarded place to measure prespecified follow-up improvements honestly.
