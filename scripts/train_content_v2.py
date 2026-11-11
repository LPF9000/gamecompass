"""Fit structured-tag/description content and ridge projection using validation only.

No test-user rankings are computed. Production integration and the separately
reserved final test are owned by the caller after configuration freeze.
"""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd

from gamecompass.content import ContentFeatures, ContentModel, atomic_tags, fit_ridge_projection, project
from gamecompass.evaluation import cold_item_partition, cold_item_queries, interaction_matrix, prepare, ranking_metrics, user_partition
from gamecompass.models import atomic_json, sha256


def summarize(queries: list, scores: np.ndarray, candidate_ids: list[str], seed: int = 42) -> dict:
    records, coverage = [], set()
    candidate_array = np.array(candidate_ids)
    for query, row in zip(queries, scores):
        support, seen, targets, negatives = query
        eligible = np.array([i for i, g in enumerate(candidate_ids) if g not in seen])
        ranking = eligible[np.lexsort((candidate_array[eligible], -np.asarray(row)[eligible]))[:20]]
        ids = [candidate_ids[i] for i in ranking]
        metrics = {**ranking_metrics(ids, targets, 10), **ranking_metrics(ids, targets, 20),
                   'explicit_negative_fraction@20': len(set(ids) & negatives) / max(1, len(ids))}
        records.append(metrics)
        coverage.update(ids)
    if not records:
        return {'queries': 0, 'status': 'insufficient validation queries'}
    result = {}
    rng = np.random.default_rng(seed)
    for key in records[0]:
        values = np.array([r[key] for r in records])
        draws = values[rng.integers(0, len(values), size=(200, len(values)))].mean(axis=1)
        result[key] = {'mean': float(values.mean()), 'ci95': [float(x) for x in np.quantile(draws, [.025, .975])], 'n': len(values)}
    return {'queries': len(records), 'metrics': result, 'catalog_coverage@20': len(coverage) / len(candidate_ids)}


def signed_profile_matrix(queries: list, ids: list[str]) -> sparse.csr_matrix:
    index = {g: i for i, g in enumerate(ids)}
    rr, cc, values = [], [], []
    for row, query in enumerate(queries):
        for game, weight in query[0].items():
            if game in index:
                rr.append(row); cc.append(index[game]); values.append(weight)
    return sparse.csr_matrix((np.array(values, dtype=np.float32), (rr, cc)), shape=(len(queries), len(ids)))


def catalog_queries(users: dict, warm: set[str], history: int, limit: int) -> list:
    """Warm-only support, then every later eligible positive target, not cold-conditioned."""
    queries = []
    for user in sorted(users):
        rows = users[user]
        positions = [i for i, row in enumerate(rows) if row[1] in warm]
        if len(positions) < history:
            continue
        prefix = positions[:history]
        later = rows[prefix[-1] + 1:]
        targets = {row[1] for row in later if row[2]}
        if not targets:
            continue
        support = {rows[i][1]: 1.0 if rows[i][2] else -1.0 for i in prefix}
        queries.append((support, set(support), targets, {r[1] for r in later if not r[2]}))
        if len(queries) >= limit:
            break
    return queries


def train(data_dir: Path, max_users: int = 300, catalog_users: int = 100, seed: int = 42) -> dict:
    started = time.monotonic()
    source_hash = sha256(data_dir / 'population.sqlite')
    current = json.loads((data_dir / 'models' / 'current.json').read_text())['version']
    reference = json.loads((data_dir / 'models' / current / 'manifest.json').read_text())
    if reference['source_sha256'] != source_hash:
        raise ValueError('Population source does not match the documented base split')
    print('Preparing seed-42 train and validation users; test users will not be scored', flush=True)
    data = prepare(data_dir, max_items=reference['max_items'], seed=seed)
    data.pop('test_users', None)
    cold = sorted(g for g in data['cf_ids'] if cold_item_partition(g, seed))
    cold_set = set(cold)
    warm = [g for g in data['cf_ids'] if g not in cold_set]
    warm_set = set(warm)
    rows = [r for r in data['train_rows'] if r[1] in warm_set]
    if any(user_partition(r[0], seed) != 'train' or r[1] in cold_set for r in rows):
        raise AssertionError('User/item holdout contamination')
    x = interaction_matrix(rows, warm)
    print(f'Fresh warm CF fit: {x.shape[0]:,} train users; {x.nnz:,} positives; {len(warm)} warm / {len(cold)} withheld items', flush=True)
    _, _, vt = randomized_svd(x, n_components=min(64, min(x.shape) - 1), random_state=seed)
    factors = normalize(vt.T, norm='l2').astype(np.float32)
    games = data['games']
    index = {g['game_id']: i for i, g in enumerate(games)}
    features = ContentFeatures.fit(games)
    tags, words = features.transform(games)
    wi = [index[g] for g in warm]
    ci = [index[g] for g in cold]
    queries_by_history = {h: cold_item_queries(data['validation_users'], cold_set, h, max_users) for h in (1, 3, 5)}
    queries = [q for h in (1, 3, 5) for q in queries_by_history[h]]
    if not queries:
        raise ValueError('No validation cold-item queries')
    support = signed_profile_matrix(queries, warm)
    latent_users = normalize(np.asarray(support @ factors), norm='l2').astype(np.float32)
    cosine_trials, ridge_trials = [], []
    best_cosine, best_projection = None, None
    print(f'Observed {len(features.tag_vocabulary)} atomic tags and {len(features.description_vocabulary)} description features; tuning on {len(queries)} validation queries only', flush=True)
    for weight in (0., .25, .5, .75, 1.):
        matrix = ContentFeatures.combine(tags, words, weight)
        warm_features, cold_features = matrix[wi], matrix[ci]
        profile = normalize(support @ warm_features, norm='l2')
        cosine_scores = (profile @ cold_features.T).toarray()
        metrics = summarize(queries, cosine_scores, cold, seed)
        cosine_trials.append({'tag_weight': weight, 'result': metrics})
        value = metrics['metrics']['ndcg@20']['mean']
        if best_cosine is None or value > best_cosine['value']:
            best_cosine = {'tag_weight': weight, 'value': value}
        for alpha in (.1, 1., 10., 100.):
            coefficients = fit_ridge_projection(warm_features, factors, alpha)
            cold_factors = project(cold_features, coefficients)
            scores = latent_users @ cold_factors.T
            result = summarize(queries, scores, cold, seed)
            ridge_trials.append({'tag_weight': weight, 'alpha': alpha, 'result': result})
            value = result['metrics']['ndcg@20']['mean']
            if best_projection is None or value > best_projection['value']:
                best_projection = {'tag_weight': weight, 'alpha': alpha, 'value': value, 'coefficients': coefficients}
    model = ContentModel()
    model.features = features
    model.game_ids = [g['game_id'] for g in games]
    model.warm_ids = warm
    model.warm_factors = factors
    model.projection = best_projection['coefficients']
    model.tags, model.words = tags, words
    model.cosine_tag_weight = best_cosine['tag_weight']
    model.projection_tag_weight = best_projection['tag_weight']
    model.projection_alpha = best_projection['alpha']
    model.selected = 'projection' if best_projection['value'] > best_cosine['value'] else 'cosine'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    version = f'content-v2-validation-{stamp}'
    model.metadata = {
        'source_sha256': source_hash, 'seed': seed, 'base_cf_scope': len(data['cf_ids']),
        'cold_item_ids': cold, 'training_users': x.shape[0], 'training_positives': x.nnz,
        'validation_queries': len(queries), 'selection_metric': 'pooled validation cold-item NDCG@20 at support1,3,5',
        'test_evaluation': 'not run; final test must be reserved separately before integration',
        'source_dataset': 'Kaggle Steam recommendations v28; source user_id%32 complete-user cohort',
        'metadata_snapshot_caveat': 'Metadata snapshot may postdate historical interactions',
        'version': version,
    }
    model._indexes()
    artifact = data_dir / 'content-models' / version
    model.save(artifact)
    frozen = {'version': version, 'path': str(artifact), 'selected': model.selected,
              'cosine_tag_weight': model.cosine_tag_weight, 'projection_tag_weight': model.projection_tag_weight,
              'projection_alpha': model.projection_alpha, 'source_sha256': source_hash,
              'manifest_sha256': sha256(artifact / 'manifest.json'), 'selection_phase': 'validation only; no test scores'}
    atomic_json(artifact / 'frozen-selection.json', frozen)
    atomic_json(data_dir / 'content-models' / 'latest-validation.json', frozen)
    by_history = {}
    for history, cohort in queries_by_history.items():
        by_history[str(history)] = {
            a: summarize(cohort, np.array([model.score(q[0], cold, algorithm=a) for q in cohort]), cold, seed)
            for a in ('tags', 'description', 'cosine', 'projection')
        }
    # Broader validation: all 50k cached catalog candidates; targets come from
    # remaining labeled reviews in the 1500-item bounded evidence universe.
    print(f'Frozen configuration: {frozen}; evaluating broader catalog on validation only', flush=True)
    catalog = {}
    warm_view = np.zeros((len(model.game_ids), factors.shape[1]), dtype=np.float32)
    warm_view[wi] = factors
    for history in (1, 3, 5):
        cohort = catalog_queries(data['validation_users'], warm_set, history, catalog_users)
        profile = normalize(np.asarray(signed_profile_matrix(cohort, warm) @ factors), norm='l2')
        catalog[str(history)] = {
            'warm_cf_no_cold_representation': summarize(cohort, profile @ warm_view.T, model.game_ids, seed),
        }
        for algorithm in ('tags', 'description', 'cosine', 'projection'):
            scores = np.array([model.score(q[0], algorithm=algorithm) for q in cohort])
            catalog[str(history)][algorithm] = summarize(cohort, scores, model.game_ids, seed)
    report = {
        'phase': 'validation only; neither old nor reserved final test users scored', 'frozen': frozen,
        'catalog_games': len(games), 'tags_observed_games': sum(bool(atomic_tags(g)) for g in games),
        'tag_features': len(features.tag_vocabulary), 'description_features': len(features.description_vocabulary),
        'warm_items': len(warm), 'cold_items': len(cold), 'training_users': x.shape[0], 'training_positives': x.nnz,
        'cosine_trials': cosine_trials, 'projection_trials': ridge_trials,
        'cold_validation_by_history': by_history, 'catalog_validation_by_history': catalog,
        'catalog_query_cap_per_history': catalog_users, 'cold_query_cap_per_history': max_users,
        'duration_seconds': time.monotonic() - started,
        'limitations': [
            'Validation was previously used during exploration; it is suitable for selection, not a fresh accuracy claim',
            'No test ranking has been run in this experiment; a reserved final test is required',
            'Cold-item scope is a hash-selected subset of historically observed train-user-popular1500 games, not actual new releases',
            'Static metadata tags/descriptions may postdate historical interactions',
            'Pooled validation queries repeat users across support lengths; bootstrap intervals are descriptive',
            'Broad-catalog targets cover only later labeled positives in the bounded1500-item evidence universe; other games are unobserved distractors',
            'Source is historical Steam review feedback; console coverage and future enjoyment remain unverified',
        ],
    }
    atomic_json(data_dir / 'content-v2-validation-results.json', report)
    docs = Path(__file__).resolve().parents[1] / 'docs'
    atomic_json(docs / 'content-v2-validation-results.json', report)
    lines = ['# Structured game content v2 — validation only', '',
             'The prior content feature path omitted the real Kaggle `metadata.tags` field. This version reads those tags as atomic labels, deduplicates them, and gives descriptions a separate fitted word/bigram TF-IDF vocabulary. Tags and descriptions are normalized independently; feature blend weights are selected on validation users.', '',
             f"Catalog: {len(games):,} games; {report['tags_observed_games']:,} have structured tags. Fitted features: {report['tag_features']} atomic tags and {report['description_features']:,} description terms/bigrams.", '',
             f"Fresh SVD fit uses {x.shape[0]:,} training users and {x.nnz:,} positive interactions over {len(warm)} warm items. Every one of the {len(cold)} cold items and every validation/test user is excluded from collaborative fitting.", '',
             f"Selected cosine tag weight: {model.cosine_tag_weight}; selected ridge tag weight: {model.projection_tag_weight}, alpha: {model.projection_alpha}. Default content method: **{model.selected}**. Selection uses pooled cold-item validation NDCG@20, support lengths1/3/5; no test ranking is computed.", '',
             '## Cold-item validation', '', '| Warm support | Queries | Tags NDCG@20 | Description NDCG@20 | Selected cosine NDCG@20 | Projection NDCG@20 |', '|---|---:|---:|---:|---:|---:|']
    for history, results in by_history.items():
        cells = [f"{results[a]['metrics']['ndcg@20']['mean']:.4f}" if results[a].get('queries') else 'n/a' for a in ('tags','description','cosine','projection')]
        lines.append(f"| {history} | {results['tags']['queries']} | " + ' | '.join(cells) + ' |')
    lines += ['', '## Full-catalog validation', '', 'Candidate universe contains every cached catalog game, with support games excluded. Query targets are later positive reviews inside the bounded evidence universe. Queries are not conditioned on having a cold target. The same queries/candidates apply to every method.', '',
              '| Warm support | Queries | Warm CF only NDCG@20 | Tags | Description | Selected cosine | Projection |', '|---|---:|---:|---:|---:|---:|---:|']
    for history, results in catalog.items():
        methods = ('warm_cf_no_cold_representation','tags','description','cosine','projection')
        cells = [f"{results[a]['metrics']['ndcg@20']['mean']:.4f}" if results[a].get('queries') else 'n/a' for a in methods]
        lines.append(f"| {history} | {results['tags']['queries']} | " + ' | '.join(cells) + ' |')
    lines += ['', '## Artifact and API', '', f"Frozen artifact: `{artifact}`. Its manifest records vocabularies, feature parameters, selection, source fingerprint and component checksums. Arrays store fitted IDF, ridge coefficients and warm factors; sparse NPZ files store catalog feature blocks. No pickle or API account is needed.", '',
              '`ContentModel.load(path).score(support, candidates=None, support_games=None, algorithm=None)` returns ranking scores aligned with supplied candidate game dictionaries or canonical ID strings; omitted candidates mean the cached catalog. Algorithms: `tags`, `description`, `cosine`, `projection`. Repeated known candidate sets use cached sparse/dense views. New candidate metadata is transformed with fitted vocabularies/IDF. New support metadata can be supplied through `support_games`. Unknown/missing metadata gives zero evidence.', '',
              'For projection, warm games use their fitted collaborative factors; every cold/new game uses predicted factors. User representation is a normalized signed aggregation. Original production weights and pointer are not changed.', '', '## Limits', '']
    lines += [f'- {item}' for item in report['limitations']]
    lines += ['', f"Elapsed: {report['duration_seconds']:.1f}s. Full selection grid and metrics: content-v2-validation-results.json.", '',
              'Reproduce: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/train_content_v2.py --data-dir data --max-users 300 --catalog-users 100 --seed 42`.']
    (docs / 'content-v2.md').write_text('\n'.join(lines) + '\n')
    return report


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--max-users', type=int, default=300)
    parser.add_argument('--catalog-users', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    result = train(args.data_dir, args.max_users, args.catalog_users, args.seed)
    print(json.dumps(result['frozen'], indent=2), flush=True)


if __name__ == '__main__':
    main()
