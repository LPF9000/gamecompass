# Global chronological development benchmark

global 80th-percentile timestamp cutoff, all CF/popularity fitting at/before cutoff; targets strictly later; no tuning on temporal targets; full eligible item ranking excluding prior reviews

Cutoff: 2022-02-16T00:00:00+00:00. Timestamped rows: 1,295,681; fitting interactions: 844,952; later interactions: 259,045.
Eligible games: 1,500; queries: 300; future interaction item-scope coverage: 64.78%.

| Algorithm | Recall@20 | NDCG@20 | NDCG 95% CI |
|---|---:|---:|---|
| popularity | 0.0997 | 0.0431 | 0.0323–0.0575 |
| itemknn | 0.1256 | 0.0576 | 0.0430–0.0706 |
| ease | 0.1221 | 0.0571 | 0.0408–0.0709 |
| svd | 0.1241 | 0.0525 | 0.0385–0.0680 |
| content | 0.0702 | 0.0327 | 0.0221–0.0464 |
| hybrid | 0.1223 | 0.0572 | 0.0408–0.0708 |

This separate temporal test is not used for model selection. All CF/popularity fitting is at/before the cutoff and every target is strictly later. Content metadata is a later snapshot and has a metadata look-ahead limitation.

- Static metadata may postdate cutoff; temporal content/hybrid results have metadata look-ahead limitation
- Eligible items selected only from prior positives; future unseen items excluded and coverage reported
- First sorted eligible users capped by max_users, not random sampling

Dataset SHA-256: `e873ed8dc766549b66d69496a6580d38242fa84786f6ce029609241b94e0a76f`.

Full aggregate results: temporal-benchmark-results.json. Reproduce: `gamecompass benchmark-temporal --max-items 1500 --max-users 300 --seed 42`.
