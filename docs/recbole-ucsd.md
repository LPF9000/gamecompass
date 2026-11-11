# Historical UCSD RecBole development experiments

Source: UCSD Australian Steam research reviews; not the current production benchmark. These historical development runs use 5 epochs and at most 500 queries per cohort. Dataset/model redistribution permission remains unresolved; weights stay local.

Fixed 5 epochs; seed 42; 18,947 population users, 1,500 eligible games, 34,904 positive interactions.

Same full training-user-selected eligible item universe and complete held-out users as gamecompass benchmark; max 500 queries per cohort

Fixed epochs chosen before evaluation. No per-user embedding from validation/test exists. Signed sum of learned item embeddings produces unseen-user scores. Not probabilities.

| Model | Seconds | Validation NDCG@20 | Cold support=1 NDCG@20 | Cold support=3 NDCG@20 |
|---|---:|---:|---:|---:|
| BPR | 1.7 | 0.05890 | 0.06598 | 0.07404 |
| LightGCN | 5.8 | 0.16690 | 0.17070 | 0.18300 |

All metrics and 95% query bootstrap intervals: `data/recbole/results.json`. Full-catalog ranking within the bounded eligible universe; no sampled evaluation negatives. Fixed shallow training is a development baseline, not a tuned neural-model ceiling. Zero-history uses the same population popularity baseline. Cohorts without sufficient held-out history are reported unavailable in the recorded metrics.

Local artifacts: `data/recbole/{BPR,LightGCN}/factors.npz` (safe NumPy arrays) and own-training checkpoints under checkpoints/. UCSD-derived weights remain local pending rights clarification; no pretrained weights are committed.

## Reproduction

```sh
uv python install 3.11
uv venv .venv-recbole --python 3.11
uv pip install --python .venv-recbole/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu -r configs/recbole-requirements.lock
.venv-recbole/bin/python scripts/train_recbole.py --max-items 1500 --epochs 5 --seed 42
```

RecBole 1.2.1 uses removed SciPy DOK `_update` APIs; this separate environment pins SciPy 1.10.1 and pandas 2.2.3. Main app environment stays independent. CPU execution verified. CUDA/MPS are not tested by these experiments.
