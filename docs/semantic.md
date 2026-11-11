# Measured semantic content experiment

Model: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) at `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; official model card declares Apache-2.0.

Real population snapshot: 1,295,681 interactions / 430,864 users. SHA-256: `e873ed8dc766549b66d69496a6580d38242fa84786f6ce029609241b94e0a76f`.
Eligible candidate universe: 1,500 training-selected games; at most 300 heldout users per cohort. CPU elapsed 22.0 seconds.

Both methods use signed weighted item centroids, the same metadata fields, disjoint test users, identical chronological support prefixes, and the full identical eligible item universe with support items removed. No test-set tuning or production activation occurred. The TF-IDF method uses the native 4096-bin hashing baseline with IDF fit on static catalog metadata. MiniLM runs locally; only its initial official weights download needs network access.

| Support reviews | Users | TF-IDF Recall@20 | MiniLM Recall@20 | TF-IDF NDCG@20 | MiniLM NDCG@20 |
|---:|---:|---:|---:|---:|---:|
| 0 | 300 | 0.0085 | 0.0085 | 0.0073 | 0.0073 |
| 1 | 300 | 0.0131 | 0.0158 | 0.0109 | 0.0109 |
| 3 | 300 | 0.0228 | 0.0204 | 0.0170 | 0.0136 |
| 5 | 300 | 0.0159 | 0.0132 | 0.0121 | 0.0086 |
| 10 | 300 | 0.0308 | 0.0247 | 0.0257 | 0.0214 |
| 20 | 300 | 0.0194 | 0.0237 | 0.0235 | 0.0230 |
| 50 | 34 | 0.0228 | 0.0148 | 0.0421 | 0.0314 |
| 100 | 8 | 0.1201 | 0.0059 | 0.0908 | 0.0191 |

## Validation comparison

Pooled support lengths 1, 3 and 5; a validation user may contribute more than one query. No test results are used for choosing an encoder.

TF-IDF NDCG@20: 0.0112; MiniLM NDCG@20: 0.0141.

Zero-history content controls have zero vectors and canonical-ID tie order. They are not the population popularity baseline. Detailed Recall/NDCG/MAP@10/20, explicit-negative diagnostics and bootstrap intervals are saved locally in `data/semantic-results.json`.

## Reproduce

```bash
uv venv .venv-semantic --python 3.11
uv pip install --python .venv-semantic/bin/python torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-semantic/bin/python -r configs/semantic-requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
PYTHONPATH=src .venv-semantic/bin/python scripts/benchmark_semantic.py --data-dir data --max-items 1500 --max-users 300
```

Add `--local-files-only` after the initial successful acquisition. Existing embeddings are reused only when their content key and checksum match.

## Limits

- Static metadata can postdate historical reviews
- Pretrained text model may have seen game descriptions; pretraining corpus cannot be fully audited for game text
- Not fine-tuned on gaming preferences
- Bounded popular-item universe omits cold and long-tail catalog items
- Not a calibrated enjoyment probability
- No temporal population experiment in this script
