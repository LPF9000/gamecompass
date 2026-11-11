# Actual RecBole development experiments

Source: Kozyriev Steam v28 (publisher-declared CC0), complete-user 1/32 cohort: 1,295,681 reviews from 430,864 users. This is a bounded development experiment, not full-corpus training. The 1,500-item ranking universe is selected only from population-training positives. All main benchmark and neural methods use the same held-out users and at most 300 queries per history cohort.

Fixed 50 epochs; seed 42; 294,847 population users, 1,500 eligible games, 658,217 positive interactions.

Same full training-user-selected eligible item universe and complete held-out users as gamecompass benchmark; configured maximum queries per cohort

Fixed epochs chosen before evaluation. No per-user embedding from validation/test exists. Signed sum of learned item embeddings produces unseen-user scores. Not probabilities.

| Model | Seconds | Validation NDCG@20 | Cold support=1 NDCG@20 | Cold support=3 NDCG@20 |
|---|---:|---:|---:|---:|
| BPR | 182.4 | 0.03984 | 0.05015 | 0.03788 |
| LightGCN | 248.5 | 0.05298 | 0.07083 | 0.06068 |

All metrics and 95% query bootstrap intervals: `data/recbole/results.json`. Full-catalog ranking within the bounded eligible universe; no sampled evaluation negatives. Fixed shallow training is a development baseline, not a tuned neural-model ceiling. Zero-history uses the same population popularity baseline. Cohorts without sufficient held-out history are explicitly reported unavailable.

Local artifacts: `data/recbole/{BPR,LightGCN}/factors.npz` (safe NumPy arrays) and own-training checkpoints under checkpoints/. Check source data provenance before distribution; no pretrained weights are committed.

## Reproduction

```sh
uv python install 3.11
uv venv .venv-recbole --python 3.11
uv pip install --python .venv-recbole/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu -r configs/recbole-requirements.lock
.venv-recbole/bin/python scripts/train_recbole.py --max-items 1500 --epochs 50 --max-users 300 --seed 42
```

The Kaggle development run uses BPR batch size 32768 and LightGCN full batch size 1048576 (one graph update per epoch for 658217 positives) to bound dense embedding-optimizer and repeated full-graph costs on CPU. This is a fixed-budget baseline, not an extensively tuned setting. RecBole 1.2.1 uses removed SciPy DOK `_update` APIs; this separate environment pins SciPy 1.10.1 and pandas 2.2.3. Main app environment stays independent. Linux CPU execution verified. This lock contains the official Linux torch +cpu build; macOS needs the corresponding platform Torch 2.5.1 wheel rather than +cpu. CUDA/MPS are not tested by these experiments.

## Model selection and limitations

Pooled cold-user validation NDCG@20: BPR 0.03984248; LightGCN 0.05298483; production EASE/content hybrid 0.09601370. Both neural aggregation baselines lose validation, so the production hybrid is retained. These are fixed-budget configurations, not a claim that all BPR/LightGCN models or learned user encoders underperform. Support-length 50 and 100 cohorts contain 34 and 8 queries respectively, so their estimates are especially uncertain. Early stopping or hyperparameter search on the test set was not performed. The LightGCN full-batch setting reduces full-graph recomputation and changes optimizer update frequency relative to BPR; equal epochs are not equal update counts.

Resume and checkpoint safety: [recbole-resume.md](recbole-resume.md). Ordinary Torch-free tests: 3 passed, 1 optional training test skipped. Isolated RecBole environment: all 4 export/guard/real-resume tests passed in 4.20 seconds. The optional synthetic fixture verifies checkpoint mechanics only, never recommendation quality.

Public aggregate metrics: [recbole-results.json](recbole-results.json). Source hashes and local factor checksums are recorded there; no individual user IDs, raw records, user embeddings, or model weights are published.
