#!/usr/bin/env python3
"""Validation-only fold-in experiments. Deliberately has no test-scoring option."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
from threadpoolctl import threadpool_limits
from gamecompass.evaluation import cold_queries, evaluate_queries, prepare
from gamecompass.foldin import FoldInConfig, FoldInModel, load_item_factors
from gamecompass.models import Model, atomic_json, sha256


class Adapter:
    """Only catalog item state is exposed to the shared ranking evaluator."""
    def __init__(self, baseline: Model, scorer: FoldInModel):
        self.game_ids = baseline.game_ids
        self.cf_game_ids = baseline.cf_game_ids
        self._idx = baseline._idx
        self._cf_out = baseline._cf_out
        self.popularity = baseline.popularity
        self.content = baseline.content
        self.scorer = scorer

    def score(self, support, algorithm=None):
        return self.scorer.score(support)


def configurations():
    for normalization in ('none', 'l2', 'center_l2'):
        for method in ('sum', 'centroid'):
            yield FoldInConfig(method=method, normalization=normalization)
        for method in ('ridge', 'logistic'):
            for regularization in (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0):
                yield FoldInConfig(method=method, normalization=normalization, regularization=regularization)


def validation_queries(prepared: dict, max_users: int):
    return [query for history in (1, 3, 5)
            for query in cold_queries(prepared['validation_users'], history, max_users)]


def objective(result: dict) -> float:
    return result['metrics']['ndcg@20']['mean']


def run(data_dir: Path, max_items: int = 1500, max_users: int = 300, seed: int = 42) -> dict:
    started = time.monotonic()
    data_dir = Path(data_dir)
    active = json.loads((data_dir/'active-population.json').read_text())
    if sha256(data_dir/'population.sqlite') != active['sha256']:
        raise ValueError('Population source checksum differs from active provenance')
    pointer = json.loads((data_dir/'models/current.json').read_text())
    full_model = Model.load(data_dir/'models'/pointer['version'])
    if full_model.metadata['source_sha256'] != active['sha256']:
        raise ValueError('Baseline was trained from a different population source')
    prepared = prepare(data_dir, max_items=max_items, seed=seed)
    # Do not construct, inspect, rank or measure any test query in this experiment.
    del prepared['test_users']
    ids = prepared['cf_ids']
    if ids != full_model.cf_game_ids:
        raise ValueError('Prepared candidate scope differs from the pretrained baseline')
    model = copy.copy(full_model)
    model.game_ids = list(ids)
    model.popularity = full_model.popularity[full_model._cf_out]
    model.content = full_model.content[full_model._cf_out]
    model._indexes()
    queries = validation_queries(prepared, max_users)
    if not queries:
        raise ValueError('No eligible validation users')
    output = data_dir/'foldin-v2'
    output.mkdir(parents=True, exist_ok=True)
    report = {
        'experiment': 'observed-only-foldin-v2-validation', 'test_scored': False,
        'population_source_sha256': active['sha256'], 'seed': seed,
        'baseline_version': pointer['version'], 'eligible_items': len(ids),
        'candidate_ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        'validation_queries': len(queries), 'max_users_per_history': max_users,
        'selection_objective': 'Pooled validation NDCG@20 at support lengths 1/3/5; first sorted max_users per cohort. Same user may appear in several cohorts.',
        'candidate_strategy': 'Entire existing training-user-selected CF item universe; support items removed identically for every model. No sampled evaluation negatives.',
        'feedback_semantics': 'Only supplied observed positive/negative support labels fit user factors. Unobserved candidates never become negative examples.',
        'baselines': {}, 'trials': [],
        'limitations': ['Validation model selection is optimistic; independent reserved test evaluation is required after freezing configuration.',
                       'Neither item-factor transformations nor new-user fits access population user embeddings.',
                       'Logistic scores are uncalibrated user-specific logits, not enjoyment probabilities.',
                       'Historical review selection, deterministic source-user subsampling, metadata timing and bounded candidate scope remain limitations.'],
    }
    for name in ('popularity', 'ease', 'hybrid'):
        report['baselines'][name] = evaluate_queries(model, queries, name, seed)
    # Explicit None delegates only to the preexisting frozen history routing.
    report['baselines']['production_routed'] = evaluate_queries(model, queries, None, seed)
    train_export_sha = sha256(data_dir/'recbole/gamecompass/gamecompass.inter')
    for source_model in ('BPR', 'LightGCN'):
        factor_ids, factors, manifest = load_item_factors(data_dir/'recbole'/source_model)
        if manifest.get('source', {}).get('sha256') != active['sha256']:
            raise ValueError(f'{source_model} factors have a different source dataset')
        if manifest.get('atomic_export_sha256') != train_export_sha:
            raise ValueError(f'{source_model} factors have a different training export')
        if set(factor_ids) != set(ids):
            raise ValueError('Item factor IDs differ from the fixed candidate universe')
        mapping = {game: index for index, game in enumerate(factor_ids)}
        aligned = factors[[mapping[game] for game in ids]]
        for number, config in enumerate(configurations(), start=1):
            scorer = FoldInModel(ids, aligned, config, popularity=model.popularity)
            adapter = Adapter(model, scorer)
            phase = time.monotonic()
            result = evaluate_queries(adapter, queries, 'foldin', seed)
            trial = {'source_model': source_model, 'factor_sha256': manifest['factor_sha256'],
                     'atomic_export_sha256': train_export_sha, 'config': asdict(config), 'validation': result,
                     'seconds': time.monotonic()-phase, 'logistic_calls': scorer.logistic_calls,
                     'logistic_nonconverged': scorer.logistic_nonconverged}
            report['trials'].append(trial)
            print(json.dumps({'source_model':source_model, 'trial':number,'config':asdict(config),'validation_ndcg20':objective(result),'seconds':trial['seconds']}), flush=True)
            atomic_json(output/'validation-progress.json', report)
    best = max(report['trials'], key=lambda trial: objective(trial['validation']))
    report['selected_foldin'] = {key:value for key,value in best.items() if key != 'validation'}
    report['selected_validation'] = best['validation']
    best_baseline = max(report['baselines'], key=lambda name: objective(report['baselines'][name]))
    report['best_baseline'] = best_baseline
    report['foldin_beats_baseline_validation'] = objective(best['validation']) > objective(report['baselines'][best_baseline])
    factor_ids, factors, _ = load_item_factors(data_dir/'recbole'/best['source_model'])
    mapping = {game:index for index,game in enumerate(factor_ids)}
    scorer = FoldInModel(ids, factors[[mapping[g] for g in ids]], FoldInConfig(**best['config']), model.popularity)
    report['selected_validation_by_history'] = {
        str(history):evaluate_queries(Adapter(model,scorer),cold_queries(prepared['validation_users'],history,max_users),'foldin',seed)
        for history in (0,1,3,5,10,20,50,100)
    }
    frozen = {'format_version':1,'selection_phase':'validation_only','test_scored':False,
              'source_model':best['source_model'],'config':best['config'],
              'population_source_sha256':active['sha256'],'factor_sha256':best['factor_sha256'],
              'atomic_export_sha256':train_export_sha,'candidate_ids_sha256':report['candidate_ids_sha256'],
              'cf_game_ids':ids,'seed':seed,'validation_objective':report['selection_objective']}
    frozen['score_semantics']='uncalibrated ranking scores, not enjoyment probabilities'
    atomic_json(output/'frozen-config.json',frozen)
    report['frozen_config_sha256'] = sha256(output/'frozen-config.json')
    report['seconds'] = time.monotonic()-started
    atomic_json(output/'validation-results.json',report)
    write_report(report,Path(__file__).resolve().parents[1]/'docs/foldin-v2.md')
    return report


def write_report(report: dict, path: Path):
    best=report['selected_foldin']
    lines=['# Validation-only observed-feedback fold-in experiment','',
           'No test users were scored. This is candidate selection, not an independent generalization result. An independent evaluation must use reserved users after freezing this configuration.','',
           f"Source population SHA-256: `{report['population_source_sha256']}`.",
           f"Candidate universe: {report['eligible_items']:,}; pooled support lengths 1/3/5 validation queries: {report['validation_queries']:,}; max {report['max_users_per_history']} per history cohort.",'',
           report['feedback_semantics'],'',report['candidate_strategy'],'',
           '| Baseline | Validation NDCG@20 | Recall@20 |','|---|---:|---:|']
    for name,result in report['baselines'].items():
        lines.append(f"| {name} | {objective(result):.6f} | {result['metrics']['recall@20']['mean']:.6f} |")
    lines += ['','## Best fold-in candidates','','| Factor model | User method | Item normalization | Regularization | Validation NDCG@20 | Recall@20 |','|---|---|---|---:|---:|---:|']
    for trial in sorted(report['trials'],key=lambda t:objective(t['validation']),reverse=True)[:12]:
        c=trial['config'];r=trial['validation']
        lines.append(f"| {trial['source_model']} | {c['method']} | {c['normalization']} | {c['regularization']:g} | {objective(r):.6f} | {r['metrics']['recall@20']['mean']:.6f} |")
    lines += ['','## Frozen configuration','',f"`{json.dumps({'source_model':best['source_model'],'config':best['config']},sort_keys=True)}`",'',
              f"Frozen JSON SHA-256: `{report['frozen_config_sha256']}`. Local artifact: `data/foldin-v2/frozen-config.json`; full grid/results: `data/foldin-v2/validation-results.json`.",'',
              'Methods: original signed sum; separately averaged positive and negative centroids; ridge regression of signed observed labels with magnitude confidence; convex regularized logistic regression of those same observed labels. Transformations are raw item factors, item L2 normalization, or training-item mean centering followed by L2 normalization. Regularization grid: 0.0001/0.001/0.01/0.1/1/10/100. Logistic optimization uses at most 100 L-BFGS iterations; convergence counts are recorded per trial. No unknown interaction is fitted as a negative. All methods accept only item IDs/factors plus the supplied support, never pretrained user IDs/vectors.','',
              'The grid is selected on pooled validation NDCG@20 only. Bootstrap intervals remain descriptive because users can repeat across support cohorts and model selection is performed on this same validation set. The old inspected test results are not used in this experiment.','',
              'Reproduce: `OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/benchmark_foldin_v2.py --max-items 1500 --max-users 300 --seed 42`.','']
    lines += ['', f"Best observed-only fold-in validation NDCG@20: {objective(report['selected_validation']):.6f}; strongest existing baseline ({report['best_baseline']}): {objective(report['baselines'][report['best_baseline']]):.6f}. The tested fold-in alternatives do not justify replacing the production baseline. This is a validation decision, not a fresh-test result or a claim about every neural model.", '']
    lines.extend(f'- {limitation}' for limitation in report['limitations'])
    path.write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data'))
    parser.add_argument('--max-items',type=int,default=1500)
    parser.add_argument('--max-users',type=int,default=300)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if args.max_items<2 or args.max_users<1:parser.error('Need max-items>=2 and max-users>=1')
    with threadpool_limits(limits=1):
        result=run(args.data_dir,args.max_items,args.max_users,args.seed)
    print(json.dumps({'selected':result['selected_foldin'],'validation_ndcg20':objective(result['selected_validation']),'test_scored':False}),flush=True)


if __name__=='__main__':main()
