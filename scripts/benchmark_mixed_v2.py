"""Validation-only mixed explicit/library-proxy selection; never scores test users."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
from gamecompass.evaluation import ranking_metrics, user_partition
from gamecompass.mixed import MixedPersonalizer
from gamecompass.models import Model, atomic_json, sha256

COEFFICIENTS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
WEIGHTINGS = ('uniform', 'normalized_log1p_hours')
ALGORITHMS = ('ease', 'hybrid')
GROUPS = tuple((n, k) for n in (5, 10, 20) for k in (1, 3))


def validation_queries(source: Path, cf_ids: list[str], *, max_users: int = 100,
                       seed: int = 42) -> tuple[dict[str, list[dict]], dict]:
    """Retrieve labels only for original validation users, never original test users."""
    if max_users < 1 or seed != 42:
        raise ValueError('Use positive cap and the original seed42 validation split')
    groups: dict[str, list[dict]] = {f'{n}:{k}': [] for n, k in GROUPS}
    histories: dict[str, list[tuple]] = {}
    with sqlite3.connect(f'file:{source.resolve()}?mode=ro', uri=True) as db:
        source_users = [r[0] for r in db.execute('SELECT DISTINCT user_id FROM interactions')]
        selected = [u for u in source_users if user_partition(u, seed) == 'validation']
        db.execute('CREATE TEMP TABLE validation_ids(user_id TEXT PRIMARY KEY)')
        db.executemany('INSERT INTO validation_ids VALUES(?)', [(u,) for u in selected])
        db.execute('CREATE TEMP TABLE eligible_items(game_id TEXT PRIMARY KEY)')
        db.executemany('INSERT INTO eligible_items VALUES(?)', [(g,) for g in cf_ids])
        for user, game, positive, timestamp, hours in db.execute('''
            SELECT i.user_id,i.game_id,i.positive,COALESCE(i.timestamp,0),COALESCE(i.playtime,0)
            FROM interactions AS i JOIN validation_ids AS v ON i.user_id=v.user_id
            JOIN eligible_items AS e ON i.game_id=e.game_id
            ORDER BY i.user_id,COALESCE(i.timestamp,0),i.game_id'''):
            histories.setdefault(user, []).append((game, int(positive), int(timestamp), float(hours)))
        source_interactions = db.execute('SELECT COUNT(*) FROM interactions').fetchone()[0]
    for n, k in GROUPS:
        cohort = groups[f'{n}:{k}']
        for user in sorted(histories):
            rows = histories[user]
            if len(rows) <= n:
                continue
            target = {row[0] for row in rows[n:] if row[1]}
            if not target:
                continue
            support = rows[:n]
            cohort.append({'user_id':user,
                           'explicit':{g:1.0 if positive else -1.0 for g,positive,_,_ in support[:k]},
                           'implicit_hours':{g:hours for g,_,_,hours in support},
                           'seen':{g for g,_,_,_ in support},'target':target,
                           'negative_targets':{row[0] for row in rows[n:] if not row[1]}})
            if len(cohort) == max_users:
                break
    stats = {'source_users':len(source_users),'source_interactions':source_interactions,
             'original_validation_users':len(selected),'validation_users_with_eligible_items':len(histories),
             'cohort_queries':{key:len(value) for key,value in groups.items()}}
    return groups, stats


def _metrics(model: Model, query: dict, support: dict[str, float], algorithm: str | None) -> dict:
    scores = model.score(support, algorithm)
    candidates = np.array([i for i,g in enumerate(model.game_ids) if g not in query['seen']])
    order = candidates[np.lexsort((np.array(model.game_ids)[candidates],-scores[candidates]))[:20]]
    ranking = [model.game_ids[i] for i in order]
    return {**ranking_metrics(ranking,query['target'],10),**ranking_metrics(ranking,query['target'],20),
            'explicit_negative_fraction@20':len(set(ranking)&query['negative_targets'])/max(1,len(ranking))}


def summarize(records: list[dict], *, seed: int = 42) -> dict:
    if not records:
        return {'queries':0,'status':'insufficient history'}
    rng=np.random.default_rng(seed)
    result={}
    for key in records[0]:
        values=np.array([r[key] for r in records])
        boots=values[rng.integers(0,len(values),size=(200,len(values)))].mean(axis=1)
        result[key]={'mean':float(values.mean()),'ci95':np.quantile(boots,[.025,.975]).tolist(),'n':len(values)}
    return {'queries':len(records),'metrics':result}


def _implicit_route(model: Model, count: int) -> dict:
    config=model.metadata.get('implicit')
    if not config:
        raise ValueError('Base artifact lacks validated implicit-only configuration')
    buckets=[int(k) for k in config.get('selection_by_history',{}) if int(k)<=count]
    return config['selection_by_history'][str(max(buckets))] if buckets else config


def run(data_dir: Path, max_users: int = 100, seed: int = 42) -> dict:
    started=time.monotonic()
    version=json.loads((data_dir/'models'/'current.json').read_text())['version']
    model=Model.load(data_dir/'models'/version)
    source_hash=sha256(data_dir/'population.sqlite')
    if model.metadata['source_sha256'] != source_hash:
        raise ValueError('Base artifact and source snapshot differ')
    # Project to the exact training-selected1500 universe; no dense full-catalog work.
    full=model
    model=copy.copy(full)
    model.game_ids=list(full.cf_game_ids)
    model.popularity=full.popularity[full._cf_out]
    model.content=full.content[full._cf_out]
    model._indexes()
    cohorts,stats=validation_queries(data_dir/'population.sqlite',model.cf_game_ids,max_users=max_users,seed=seed)
    print(f'Validation only: {stats}',flush=True)
    all_records: dict[str,list] = {}
    by_group={}
    query_digest=hashlib.sha256()
    for group,queries in cohorts.items():
        n,k=map(int,group.split(':'))
        records: dict[str,list]={'explicit_current_route':[],'implicit_current_route':[]}
        for query in queries:
            query_digest.update((query['user_id']+':'+group+'\n').encode())
            records['explicit_current_route'].append(_metrics(model,query,query['explicit'],None))
            route=_implicit_route(model,n)
            weak={g:float(np.log1p(hours)) if route['weighting']=='log1p_hours' else 1.0
                  for g,hours in query['implicit_hours'].items()}
            records['implicit_current_route'].append(_metrics(model,query,weak,route['algorithm']))
            for algorithm in ALGORITHMS:
                for weighting in WEIGHTINGS:
                    for coefficient in COEFFICIENTS:
                        key=f'{algorithm}:{weighting}:{coefficient:g}'
                        support=MixedPersonalizer.combine(query['explicit'],query['implicit_hours'],coefficient,weighting)
                        records.setdefault(key,[]).append(_metrics(model,query,support,algorithm))
        by_group[group]={key:summarize(rows,seed=seed) for key,rows in records.items()}
        for key,rows in records.items():
            all_records.setdefault(key,[]).extend(rows)
    pooled={key:summarize(rows,seed=seed) for key,rows in all_records.items()}
    mixed_keys=[key for key in pooled if key.count(':')==2 and pooled[key]['queries']]
    if not mixed_keys:
        raise ValueError('No eligible validation queries')
    # Simpler/no-influence settings win exact ties by iteration order (coefficient0 first).
    winner=max(mixed_keys,key=lambda key:pooled[key]['metrics']['ndcg@20']['mean'])
    algorithm,weighting,coefficient=winner.split(':')
    config={'schema_version':1,'algorithm':algorithm,'weighting':weighting,'coefficient':float(coefficient),
            'base_model_version':version,'dataset_sha256':source_hash,
            'selection_objective':'pooled validation NDCG@20; total supportN5/10/20, explicitk1/3; first100 eligible validation users/group',
            'scope':'reviewed-game histories as library proxy; snapshot hours; no test scoring',
            'test_status':'not scored; awaiting prospectively frozen fresh-user protocol'}
    report={'base_model_version':version,'dataset_sha256':source_hash,'seed':seed,
            'stats':stats,'eligible_cf_items':len(model.game_ids),'max_validation_users_per_group':max_users,
            'validation_query_identity_sha256':query_digest.hexdigest(),'selected_config':config,
            'coefficient_grid':list(COEFFICIENTS),'weighting_grid':list(WEIGHTINGS),'algorithm_grid':list(ALGORITHMS),
            'validation_pooled':pooled,'validation_by_group':by_group,
            'candidate_strategy':'full identical1500-item universe excluding ALL N support games for every compared method; no sampled negatives',
            'explicit_labels':'only firstk labels visible as +1/-1; remaining support labels hidden; all positive targets afterN',
            'implicit_labels':'ownership/exposure and source playtime hours only; never persisted as enjoyment ratings',
            'confidence_intervals':'200 query bootstrap replicates; pooled groups repeat users, so pooled CIs are descriptive, not independent-user inference',
            'test_scoring':False,'duration_seconds':time.monotonic()-started,
            'limitations':['Reviewed games are an exposure/review-selected library proxy, not a random sample of owned games',
                           'Source snapshot hours can include playtime after the review timestamp; this is not a temporal implicit benchmark',
                           'Chronological order is based on review dates, not acquisition or first-play dates',
                           'Only original seed42 validation users were scored; fresh-user test must precede production activation',
                           'Repeated users across total-support/explicit-label groups are not independent pooled observations',
                           'Candidate universe is training-selected popular1500 games; cold/rare catalog items not evaluated']}
    if sha256(data_dir/'population.sqlite') != source_hash:
        raise RuntimeError('Source changed during experiment')
    atomic_json(data_dir/'mixed-v2-validation.json',report)
    atomic_json(data_dir/'mixed-v2-config.json',config)
    atomic_json(Path(__file__).resolve().parents[1]/'docs'/'mixed-v2-validation.json',report)
    write_report(report,Path(__file__).resolve().parents[1]/'docs'/'mixed-v2.md')
    return report


def write_report(report: dict, path: Path) -> None:
    picked=report['selected_config']
    key=f"{picked['algorithm']}:{picked['weighting']}:{picked['coefficient']:g}"
    lines=['# Mixed explicit and implicit validation experiment','',
           '**Validation only. No test-user predictions or production activation occurred.**','',
           f"Source: {report['stats']['source_interactions']:,} reviews / {report['stats']['source_users']:,} users. SHA-256 `{report['dataset_sha256']}`. Base model `{report['base_model_version']}`.",'',
           'Original seed42 whole-user validation partition only. Total chronological supportN is5,10 or20 eligible reviews. Only firstk=1 or3 enjoyment labels are revealed; every support ID and its source-snapshot hours are available as a weak library proxy. Positives afterN are targets. All methods rank the identical full1500-item candidate universe with ALL N observed support games removed. Hidden support labels are never passed to fold-in.','',
           'Explicit labels always override implicit evidence; explicit neutral blocks positive implicit evidence and not-interested items contribute no positive feature. Weak weights are temporary inference features, never saved ratings. Uniform weighting represents observed ownership/exposure; normalized log1p(hours) divides by the largest eligible unjudged log1p value for that user.','',
           f"Selected on pooled validation NDCG@20: **{picked['algorithm']}, {picked['weighting']}, coefficient={picked['coefficient']:g}**. Coefficients tested:0,.05,.1,.25,.5,1; algorithms EASE/hybrid. Coefficient0 is included for both algorithms so algorithm-routing gains are not attributed to additional implicit evidence.",'',
           '| Total supportN | Explicit labels k | Queries | Current explicit-only NDCG@20 | Current implicit-only NDCG@20 | Selected mixed NDCG@20 |','|---:|---:|---:|---:|---:|---:|']
    for group,methods in report['validation_by_group'].items():
        n,k=group.split(':')
        if not methods[key]['queries']:
            lines.append(f'|{n}|{k}|0|—|—|—|');continue
        vals=[methods[m]['metrics']['ndcg@20']['mean'] for m in ('explicit_current_route','implicit_current_route',key)]
        lines.append(f"|{n}|{k}|{methods[key]['queries']}|"+'|'.join(f'{v:.4f}' for v in vals)+'|')
    lines+=['','## Pooled validation','', '| Method | Queries | NDCG@20 | Recall@20 |','|---|---:|---:|---:|']
    for name in ('explicit_current_route','implicit_current_route',key):
        result=report['validation_pooled'][name]
        lines.append(f"|{name}|{result['queries']}|{result['metrics']['ndcg@20']['mean']:.4f}|{result['metrics']['recall@20']['mean']:.4f}|")
    lines+=['','## Limits','']+['- '+x for x in report['limitations']]
    lines+=['','Detailed aggregate metrics/grid: `mixed-v2-validation.json`. Local selection config: `data/mixed-v2-config.json`. Keep this configuration fixed before the fresh-user test.','',
            'Reproduce: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/benchmark_mixed_v2.py --data-dir data --max-users 100`.']
    path.write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data'))
    parser.add_argument('--max-users',type=int,default=100)
    args=parser.parse_args()
    result=run(args.data_dir,args.max_users)
    print(json.dumps(result['selected_config'],indent=2))
