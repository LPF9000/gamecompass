# Training, evaluation and model updates

The standard application uses Python 3.12 and portable NPZ/JSON artifacts. RecBole experiments use an isolated Python 3.11 environment because of its older dependency APIs. See `docs/recbole.md` for verified commands. Semantic embeddings are an optional independent experiment, not a mandatory hosted service.

## Modes

| Mode | Dataset scope | Item scope | Purpose |
|---|---|---|---|
| Smoke | Real acquired data, same split | 100 training-selected items | Validate pipeline/compatibility, not quality |
| Development | Complete histories for deterministic 1/32 users | 1500 training-selected items | Reproducible bounded comparisons |
| Full acquisition | Complete histories for all users, `--user-modulus 1` | CF currently capped at 6000 | Larger experiment requiring separate resource planning; not measured here |

`gamecompass benchmark --max-items 1500 --seed 42` trains popularity, item-kNN, EASE, SVD and content models, tunes hybrid parameters on validation users and writes an experiment plus a versioned model. All models use the same split and eligible candidate universe. The experiment records source/split hashes and actual metrics. Do not describe this as a full-catalog benchmark: the production catalog can be larger than the evaluation universe.

`gamecompass temporal-benchmark --max-items 1500` evaluates a global timestamp cutoff independently of the user-holdout experiment. This does not guarantee metadata was available at prediction time; current static content is a documented limitation.

## Immediate updates

Each feedback change updates signed fold-in on the next recommendation request. No global training or data transfer is required. Playtime/ownership remain distinct implicit signals; importing a library never silently marks games loved. With no explicit preferences, a separately validated implicit adapter uses observed-game or log1p(hours) weights; reviewed-game histories are only a proxy for full libraries. Explicit ratings take precedence over implicit signals because mixed confidence weights remain unvalidated. See docs/benchmarks-implicit.md.

## Global retraining and rollback

Training always uses explicit local input files. To incorporate your own local profiles intentionally, create a separate snapshot:

```bash
gamecompass --data-dir data feedback-snapshot private-training-v1
gamecompass --data-dir private-training-v1 benchmark --max-items 1500
```

That snapshot contains personal preference labels and must not be published. Default population data and production model are unaffected by snapshot creation. Train/evaluate the candidate, inspect validation evidence, and retain previous versions. Explicit activation after artifact integrity checks:

```bash
gamecompass --data-dir data activate-model VERSION_DIRECTORY_NAME
```

The registry also supports a minimum-validation-NDCG gate programmatically. Activation history records previous versions. Benchmark runs produce completed artifacts before switching the pointer; failed/interrupted runs should not become active. Training is manually invoked, not automatically scheduled. Neural research scripts save checkpoints; closed-form baselines restart their bounded fit if interrupted. Automatic mid-fit resumption is not implemented for every algorithm.
