"""Fit a finite multi-channel ranking search on validation users only."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sqlite3
import time
import numpy as np
from gamecompass.blending import fit_blend,normalized_channels,validation_objective
from gamecompass.evaluation import cold_queries
from gamecompass.models import Model,atomic_json,sha256
from gamecompass.quality import QualityPrior

NAMES=['baseline','tags','description','projection','approval','itemknn','logpopularity']


def cf_view(full):
    model=copy.copy(full);model.game_ids=list(full.cf_game_ids)
    model.content=full.content[full._cf_out];model.popularity=full.popularity[full._cf_out];model._indexes()
    return model


def evidence_channels(base,content,quality,support,games):
    n=len(games)
    return {'baseline':base.score(support),'tags':content.score(support,candidates=games,algorithm='tags'),
            'description':content.score(support,candidates=games,algorithm='description'),
            'projection':content.score(support,candidates=games,algorithm='projection'),
            'approval':quality.channels(base.game_ids,strength=20)['centered_log_odds'],
            'itemknn':base.score(support,'itemknn'),
            'logpopularity':np.log1p(base.popularity)}


def train(data_dir:Path,content_path:Path,max_users:int=150)->dict:
    from gamecompass.content import ContentModel
    started=time.monotonic();version=json.loads((data_dir/'models/current.json').read_text())['version']
    full=Model.load(data_dir/'models'/version);base=cf_view(full)
    prepared=json.loads((data_dir/'experiments'/version/'split.json').read_text())
    if sha256(data_dir/'population.sqlite') != full.metadata['source_sha256']:raise ValueError('Population source mismatch')
    with sqlite3.connect(data_dir/'population.sqlite') as db:
        db.row_factory=sqlite3.Row;by_id={r['game_id']:dict(r) for r in db.execute('SELECT * FROM games')}
    games=[by_id[g] for g in base.game_ids]
    content=ContentModel.load(content_path);quality=QualityPrior.load(data_dir/'quality/prior')
    queries=[q for n in (1,3,5) for q in cold_queries(prepared['validation_users'],n,max_users)]
    features=[];target=np.zeros((len(queries),len(games)),dtype=bool);allowed=np.ones_like(target)
    for row,(support,seen,relevant,_) in enumerate(queries):
        channels=evidence_channels(base,content,quality,support,games)
        features.append(normalized_channels(channels,NAMES))
        for g in relevant:target[row,base._idx[g]]=True
        for g in seen:allowed[row,base._idx[g]]=False
        if row%100==0:print(f'Validation features {row}/{len(queries)}',flush=True)
    features=np.array(features,dtype=np.float32)
    selected,trials=fit_blend(NAMES,features,target,allowed,base.game_ids)
    baseline=validation_objective(features,np.eye(len(NAMES))[0],target,allowed,base.game_ids)
    gain=selected.metadata['validation_ndcg@20']-baseline
    # Predeclared simplicity safeguard, unrelated to any confirmation-test scores.
    promote=bool(gain>=.001)
    selected.metadata.update({'base_version':version,'source_sha256':full.metadata['source_sha256'],
         'content_manifest_sha256':sha256(content_path/'manifest.json'),'approval_strength':20,
         'validation_queries':len(queries),'baseline_validation_ndcg@20':baseline,
         'minimum_validation_gain_for_default':.001,'validation_gain':gain,'default_promotion_approved':promote,
         'normalization_reference':'original CF item universe before filtering/retrieval',
         'retrieval_per_channel':200})
    destination=data_dir/'blend-v2';destination.mkdir(exist_ok=True)
    selected.save(destination/'blend.json')
    report={'config':{**selected.metadata,'names':NAMES,'weights':selected.weights.tolist()},'trials':trials,
            'duration_seconds':time.monotonic()-started,'test_scored':False,
            'limitations':['Repeated validation users; this is tuning evidence, not independent confirmation',
              'Candidate item universe limited to1500 training-selected games','Review/exposure bias and historical metadata snapshot'],
            'content_artifact':str(content_path)}
    atomic_json(destination/'validation.json',report)
    atomic_json(Path('docs/blend-v2-validation.json'),report)
    weights=', '.join(f'{n}={w:.3f}' for n,w in zip(NAMES,selected.weights) if w)
    Path('docs/blend-v2.md').write_text(f'''# Validation-selected evidence combination

The existing collaborative ranking, atomic tags, descriptions, supervised projection, aggregate review approval, item neighbors and log popularity are separate channels. Weights are chosen using validation NDCG@20 across a finite, predeclared proposal grid. No fresh-test outcomes are read.

Validation queries: {len(queries)}. Baseline NDCG@20: {baseline:.6f}. Best combination: {selected.metadata['validation_ndcg@20']:.6f}. Weights: {weights}.

Default promotion requires an absolute validation gain of at least0.001, fixed before confirmation. Approved: {promote}. Repeated validation queries are not independent evidence of generalization. Full trials: blend-v2-validation.json.

Score normalization uses the same original CF reference universe offline and during serving, before platform/seen filtering. Candidate generation unions top200 per evidence channel; reranking uses the validated combination. Fresh confirmation reports retrieval agreement separately from ranking accuracy. Approval is aggregate observed review opinion, not an enjoyment probability for an individual.
''')
    print(json.dumps(report['config'],indent=2),flush=True);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,default=Path('data'))
    p.add_argument('--content-path',type=Path,required=True);p.add_argument('--max-users',type=int,default=150)
    a=p.parse_args();train(a.data_dir,a.content_path,a.max_users)
