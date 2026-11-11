"""Supervised metadata-to-CF cold-item projection; original production untouched.

All cold items are removed before collaborative fitting. Alpha selection uses
held-out validation users only, never test users or their targets.
"""
from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import time
from typing import Mapping

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd

from gamecompass.evaluation import (
    cold_item_partition, cold_item_queries, interaction_matrix, prepare,
    ranking_metrics, user_partition,
)
from gamecompass.models import Model, atomic_json, sha256


def fit_projection(warm_features: sparse.csr_matrix, warm_factors: np.ndarray,
                   cold_features: sparse.csr_matrix, alpha: float) -> np.ndarray:
    if alpha <= 0 or warm_features.shape[0] != warm_factors.shape[0]:
        raise ValueError('Positive alpha and aligned warm rows required')
    if warm_features.shape[1] != cold_features.shape[1]:
        raise ValueError('Warm and cold metadata must have identical feature dimensions')
    ridge = Ridge(alpha=alpha, solver='lsqr', fit_intercept=False, tol=1e-5)
    ridge.fit(warm_features, warm_factors)
    prediction = ridge.predict(cold_features)
    return normalize(np.asarray(prediction), norm='l2').astype(np.float32)


def user_factor(support: Mapping[str, float], warm_index: dict, factors: np.ndarray) -> np.ndarray:
    entries = [(warm_index[g], w) for g, w in support.items() if g in warm_index and w]
    if not entries:
        return np.zeros(factors.shape[1], dtype=np.float32)
    indices, weights = zip(*entries)
    profile = np.array(weights, dtype=np.float32) @ factors[list(indices)]
    length = np.linalg.norm(profile)
    return profile / length if length else profile


def summarize_rankings(queries: list, score_rows: list[np.ndarray], candidates: list[str], seed: int = 42) -> dict:
    records = []
    coverage = set()
    for query, scores in zip(queries, score_rows):
        target, negatives = query[2], query[3]
        order = np.lexsort((np.array(candidates), -np.asarray(scores)))[:20]
        ranking = [candidates[i] for i in order]
        records.append({**ranking_metrics(ranking, target, 10), **ranking_metrics(ranking, target, 20),
                        'explicit_negative_fraction@20': len(set(ranking) & negatives) / min(20, len(candidates))})
        coverage.update(ranking)
    if not records:
        return {'users': 0, 'status': 'insufficient cold-positive target queries'}
    metrics = {}
    rng = np.random.default_rng(seed)
    for key in records[0]:
        values = np.array([row[key] for row in records])
        means = values[rng.integers(0, len(values), size=(200, len(values)))].mean(axis=1)
        metrics[key] = {'mean': float(values.mean()), 'ci95': [float(v) for v in np.quantile(means, [.025, .975])], 'n': len(values)}
    return {'users': len(records), 'metrics': metrics, 'catalog_coverage@20': len(coverage) / len(candidates)}


def run_experiment(data_dir: Path, max_users: int = 300, seed: int = 42) -> dict:
    started = time.monotonic()
    current = json.loads((data_dir / 'models' / 'current.json').read_text())['version']
    cached = Model.load(data_dir / 'models' / current)
    source_hash = sha256(data_dir / 'population.sqlite')
    if source_hash != cached.metadata['source_sha256']:
        raise ValueError('Active catalog and population differ from cached artifact')
    print('Preparing training-only users and item-disjoint scope', flush=True)
    prepared = prepare(data_dir, max_items=cached.metadata['max_items'], seed=seed)
    if prepared['cf_ids'] != cached.cf_game_ids:
        raise ValueError('Prepared scope differs; use the original source/seed/configuration')
    cold_ids = sorted(g for g in prepared['cf_ids'] if cold_item_partition(g, seed))
    warm_ids = [g for g in prepared['cf_ids'] if g not in set(cold_ids)]
    warm_set = set(warm_ids)
    rows = [row for row in prepared['train_rows'] if row[1] in warm_set]
    if any(user_partition(row[0], seed) != 'train' for row in rows):
        raise AssertionError('Held-out users leaked into CF fitting')
    if any(row[1] in cold_ids for row in rows):
        raise AssertionError('Cold item leaked into CF fitting')
    matrix = interaction_matrix(rows, warm_ids)
    components = min(64, min(matrix.shape) - 1)
    print(f'Fitting fresh SVD: {matrix.shape[0]:,} training users; {len(warm_ids)} warm items; {matrix.nnz:,} positives; {len(cold_ids)} entirely removed cold items', flush=True)
    _, _, vt = randomized_svd(matrix, n_components=components, random_state=seed)
    factors = normalize(vt.T, norm='l2').astype(np.float32)
    warm_index = {g: i for i, g in enumerate(warm_ids)}
    warm_features = cached.content[[cached._idx[g] for g in warm_ids]]
    cold_features = cached.content[[cached._idx[g] for g in cold_ids]]
    queries = {split: {history: cold_item_queries(prepared[f'{split}_users'], set(cold_ids), history, max_users)
                       for history in (1, 3, 5)} for split in ('validation', 'test')}
    pooled_val = [q for history in (1, 3, 5) for q in queries['validation'][history]]
    if not pooled_val:
        raise ValueError('Insufficient validation cold targets')
    val_users = np.array([user_factor(q[0], warm_index, factors) for q in pooled_val])
    validation = {}
    predictions = {}
    print('Selecting ridge alpha on validation cold targets only', flush=True)
    for alpha in (.1, 1.0, 10.0, 100.0):
        predictions[alpha] = fit_projection(warm_features, factors, cold_features, alpha)
        scores = val_users @ predictions[alpha].T
        validation[str(alpha)] = summarize_rankings(pooled_val, list(scores), cold_ids, seed)
    best_alpha = max(predictions, key=lambda a: validation[str(a)]['metrics']['ndcg@20']['mean'])
    def content_scores(query):
        support = query[0]
        indices = [cached._idx[g] for g in support]
        center = np.asarray(cached.content[indices].T @ np.array(list(support.values()), dtype=np.float32)).ravel()
        length = np.linalg.norm(center)
        return np.asarray(cold_features @ (center / length if length else center)).ravel()
    validation_content = summarize_rankings(pooled_val, [content_scores(q) for q in pooled_val], cold_ids, seed)
    print(f'Validation-selected alpha={best_alpha}; evaluating untouched test users', flush=True)
    test = {}
    for history in (1, 3, 5):
        cohort = queries['test'][history]
        if cohort:
            profiles = np.array([user_factor(q[0], warm_index, factors) for q in cohort])
            projected = list(profiles @ predictions[best_alpha].T)
        else:
            projected = []
        test[str(history)] = {
            'projection': summarize_rankings(cohort, projected, cold_ids, seed),
            'tfidf_content': summarize_rankings(cohort, [content_scores(q) for q in cohort], cold_ids, seed),
            'zero_evidence_uniform': summarize_rankings(cohort, [np.zeros(len(cold_ids)) for _ in cohort], cold_ids, seed),
        }
    report = {
        'source_model_version_for_metadata_only': current, 'dataset_sha256': source_hash, 'seed': seed,
        'max_users_per_cohort': max_users, 'warm_items': len(warm_ids), 'cold_items': len(cold_ids),
        'cf_training_users': matrix.shape[0], 'cf_training_positives': matrix.nnz, 'latent_dimensions': components,
        'selected_alpha': best_alpha, 'validation_grid': validation, 'validation_tfidf': validation_content, 'cold_test': test,
        'selection_objective': 'NDCG@20 pooled validation cold-item queries at warm-support lengths1,3,5; repeated users may occur across pooled queries',
        'protocol': f'The experiment uses the same {len(cold_ids)} games selected by fixed item hashes and the same held-out test-user cohorts as the cold-item evidence ablation. A fresh SVD is fitted on warm-item interactions from training users. Ridge regression maps metadata TF-IDF to normalized warm latent factors and predicts normalized cold latent factors. Ranking uses signed warm-user aggregation over every cold candidate. Alpha is selected using validation users only.',
        'duration_seconds': time.monotonic() - started,
        'production_integration': 'not integrated; experiment does not change model weights or production pointer',
        'limitations': [
            'Candidate scope was selected from train-user-positive counts before item withholding, so cold candidates are historically observed popular games, not actual new releases',
            'Metadata snapshot may postdate review times; this is not a release-date temporal experiment',
            'Queries are conditional on a later positive cold target and first sorted capped users; not all-user satisfaction estimates',
            'Single seeded split with descriptive user-bootstrap intervals; pooled validation queries repeat some users',
            'SVD normalization and ridge alpha grid are fixed modeling choices; no evidence of universal optimality',
            'Zero evidence baseline is a canonical tie ordering; other constant tie orders can change that baseline',
        ],
    }
    atomic_json(data_dir / 'content-projection-results.json', report)
    docs = Path(__file__).resolve().parents[1] / 'docs'
    atomic_json(docs / 'content-projection-results.json', report)
    lines = ['# Supervised content-to-collaborative cold-item experiment', '', report['protocol'], '',
             f"Fresh collaborative fit: {matrix.shape[0]:,} training users, {matrix.nnz:,} positive warm-item interactions, {len(warm_ids)} warm items, {len(cold_ids)} cold items wholly excluded. Latent dimensions: {components}. Elapsed: {report['duration_seconds']:.1f}s.", '',
             'No population validation/test user or cold-item interaction enters SVD or ridge fitting. The prior production model supplies only unsupervised metadata features and item IDs; its CF weights and popularity counts are never used.', '',
             '## Validation-only alpha selection', '', '| Ridge alpha | Pooled validation NDCG@20 |', '|---|---:|']
    for alpha, result in validation.items():
        lines.append(f"| {alpha} | {result['metrics']['ndcg@20']['mean']:.4f} |")
    lines += ['', f"Selected alpha: **{best_alpha}**. Plain TF-IDF validation NDCG@20: {validation_content['metrics']['ndcg@20']['mean']:.4f}.", '',
              '## Untouched test users', '', '| Warm support | Queries | Projection NDCG@20 (95% CI) | Projection Recall@20 | TF-IDF NDCG@20 | Zero-evidence NDCG@20 |', '|---|---:|---|---:|---:|---:|']
    for history, result in test.items():
        picked = result['projection']
        if not picked['users']:
            lines.append(f'| {history} | 0 | n/a | n/a | n/a | n/a |')
            continue
        metric = picked['metrics']['ndcg@20']
        lines.append(f"| {history} | {picked['users']} | {metric['mean']:.4f} ({metric['ci95'][0]:.4f}–{metric['ci95'][1]:.4f}) | {picked['metrics']['recall@20']['mean']:.4f} | {result['tfidf_content']['metrics']['ndcg@20']['mean']:.4f} | {result['zero_evidence_uniform']['metrics']['ndcg@20']['mean']:.4f} |")
    lines += ['', '## Limits', ''] + [f'- {item}' for item in report['limitations']]
    lines += ['', report['production_integration'], '', 'Full aggregate results: content-projection-results.json.', '', 'Reproduce: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/benchmark_content_projection.py --data-dir data --max-users 300 --seed 42`.']
    (docs / 'content-projection.md').write_text('\n'.join(lines) + '\n')
    return report


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--max-users', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    result = run_experiment(args.data_dir, args.max_users, args.seed)
    print(f"Selected alpha {result['selected_alpha']}; report docs/content-projection.md", flush=True)


if __name__ == '__main__':
    main()
