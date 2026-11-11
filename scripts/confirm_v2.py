"""Score prospectively frozen, previously uninspected users; never tune on them."""
from __future__ import annotations
import argparse,json,sqlite3,time
from pathlib import Path
import numpy as np
from gamecompass.models import Model,atomic_json,sha256
from gamecompass.content import ContentModel
from gamecompass.quality import QualityPrior
from gamecompass.mixed import MixedPersonalizer
from gamecompass.foldin import FoldInConfig,FoldInModel,load_item_factors
from gamecompass.blending import ScoreBlend,CandidateGenerator,normalized_channels
from gamecompass.protocol import cohort_users,freeze_experiment_config,public_summary
from gamecompass.evaluation import cold_queries,cold_item_queries,cold_item_partition,ranking_metrics
from train_blend_v2 import cf_view,evidence_channels


def query_metrics(scores,ids,seen,targets,negative,eligible=None):
    candidates=np.array([i for i,g in enumerate(ids) if g not in seen and (eligible is None or i in eligible)],dtype=int)
    order=candidates[np.lexsort((np.asarray(ids)[candidates],-np.asarray(scores)[candidates]))[:20]]
    ranked=[ids[i] for i in order]
    return {**ranking_metrics(ranked,targets,10),**ranking_metrics(ranked,targets,20),
            'known_negative_fraction@20':len(set(ranked)&negative)/max(1,len(ranked))},set(ranked)


def summarize(records,seed=20261110):
    if not records:return {'users':0,'status':'no unexposed eligible users'}
    rng=np.random.default_rng(seed);result={}
    for key in records[0]:
        values=np.array([r[key] for r in records]);boots=values[rng.integers(0,len(values),(400,len(values)))].mean(axis=1)
        result[key]={'mean':float(values.mean()),'ci95':np.quantile(boots,[.025,.975]).tolist()}
    return {'users':len(records),'metrics':result}


def paired(candidate,control):
    values=np.array([a['ndcg@20']-b['ndcg@20'] for a,b in zip(candidate,control)])
    if not len(values):return None
    rng=np.random.default_rng(20261110);boots=values[rng.integers(0,len(values),(400,len(values)))].mean(axis=1)
    return {'ndcg@20_delta':float(values.mean()),'ci95':np.quantile(boots,[.025,.975]).tolist(),'users':len(values)}


def confirm(data_dir:Path,content_path:Path):
    started=time.monotonic();protocol=json.loads((data_dir/'protocols/fresh-users-v2.json').read_text())
    version=protocol['base_version'];full=Model.load(data_dir/'models'/version);base=cf_view(full)
    prepared=json.loads((data_dir/'experiments'/version/'split.json').read_text())
    content=ContentModel.load(content_path);quality=QualityPrior.load(data_dir/'quality/prior')
    blend=ScoreBlend.load(data_dir/'blend-v2/blend.json');mixed=json.loads((data_dir/'mixed-v2-config.json').read_text())
    neural=json.loads((data_dir/'foldin-v2/frozen-config.json').read_text())
    config={'base_version':version,'source_sha256':sha256(data_dir/'population.sqlite'),
      'content_manifest_sha256':sha256(content_path/'manifest.json'),'content_selected':content.selected,
      'blend':json.loads((data_dir/'blend-v2/blend.json').read_text()),'mixed':mixed,'neural':neural,
      'quality_prior_strength':20,'quality_mix':.05,'retrieval_depth':200,
      'prospective_default_decisions':{'global_blend':bool(blend.metadata['default_promotion_approved']),
        'structured_content_exploration':True,'mixed':True,'neural':False,
        'mixed_scope':'CF-mapped total evidence5..20, explicit nonneutral1..3; explicit labels override weak observations'},
      'candidate_strategy':'all original1500CF items excluding support; cold279 subset separately; no sampled negatives'}
    if config['source_sha256']!=protocol['dataset_sha256']:raise ValueError('Source differs from frozen protocol')
    locked=freeze_experiment_config(data_dir,'v2-confirmation',config,protocol)
    # Every compared hyperparameter is now immutable, before fresh predictions.
    ids,factors,_=load_item_factors(data_dir/'recbole'/neural['source_model'])
    idx={g:i for i,g in enumerate(ids)};factors=np.stack([factors[idx[g]] for g in base.game_ids])
    foldin=FoldInModel(base.game_ids,factors,FoldInConfig(**neural['config']),base.popularity)
    with sqlite3.connect(data_dir/'population.sqlite') as db:
        db.row_factory=sqlite3.Row;by_id={r['game_id']:dict(r) for r in db.execute('SELECT * FROM games')}
        selected_users=set(sum(protocol['cohorts'].values(),[]))
        hours={(u,g):float(h or 0) for u,g,h in db.execute('SELECT user_id,game_id,playtime FROM interactions') if u in selected_users}
    games=[by_id[g] for g in base.game_ids];normal={};cold={};mixreport={}
    cold_ids={g for g in base.game_ids if cold_item_partition(g,42)}
    cold_index={base._idx[g] for g in cold_ids}
    for history in (0,1,3,5,10,20,50,100):
        users=cohort_users(protocol,prepared,history)
        queries=cold_queries(users,history,500)
        records={a:[] for a in ('baseline','popularity','blend','blend_retrieved','quality_mix','foldin')}
        retrieval=[]
        for support,seen,target,negative in queries:
            channels=evidence_channels(base,content,quality,support,games)
            scores=blend.score(channels);pool=set(CandidateGenerator(200).generate(base.game_ids,channels,seen).tolist())
            quality_scores=.95*Model._scaled(base.score(support,'hybrid'))+.05*Model._scaled(channels['approval'])
            methods={'baseline':channels['baseline'],'popularity':base.popularity,'blend':scores,
                     'blend_retrieved':scores,'quality_mix':quality_scores,'foldin':foldin.score(support)}
            ranked={}
            for name,values in methods.items():
                result,ranked[name]=query_metrics(values,base.game_ids,seen,target,negative,pool if name=='blend_retrieved' else None)
                records[name].append(result)
            retrieval.append(len(ranked['blend']&ranked['blend_retrieved'])/max(1,len(ranked['blend'])))
        normal[str(history)]={'methods':{a:summarize(r) for a,r in records.items()},
            'paired_vs_baseline':{a:paired(r,records['baseline']) for a,r in records.items() if a!='baseline'},
            'retrieval_top20_agreement':float(np.mean(retrieval)) if retrieval else None}
        print(f'Fresh explicit history{history}: {len(queries)} users',flush=True)
        if history in (1,3,5):
            cq=cold_item_queries(users,cold_ids,history,500)
            cr={a:[] for a in ('old_description','tags','cosine','projection')}
            for support,seen,target,negative in cq:
                methods={'old_description':base.score(support,'content')}
                methods.update({a:content.score(support,candidates=games,algorithm=a) for a in ('tags','cosine','projection')})
                for name,values in methods.items():cr[name].append(query_metrics(values,base.game_ids,seen,target,negative,cold_index)[0])
            cold[str(history)]={'methods':{a:summarize(r) for a,r in cr.items()},
                'paired_vs_old_description':{a:paired(r,cr['old_description']) for a,r in cr.items() if a!='old_description'}}
        if history in (5,10,20):
            for count in (1,3):
                records={a:[] for a in ('explicit_only','implicit_only','mixed')}
                for user,rows in users.items():
                    support_rows=rows[:history];held=rows[history:];target={r[1] for r in held if r[2]}
                    if not target:continue
                    explicit={r[1]:1. if r[2] else -1. for r in support_rows[:count]}
                    observations={r[1]:hours.get((user,r[1]),0.) for r in support_rows}
                    combined=MixedPersonalizer.combine(explicit,observations,mixed['coefficient'],mixed['weighting'])
                    conf=base.metadata['implicit'];buckets=[int(k) for k in conf['selection_by_history'] if int(k)<=history]
                    route=conf['selection_by_history'][str(max(buckets))] if buckets else conf
                    weak={g:float(np.log1p(h)) if route['weighting']=='log1p_hours' else 1. for g,h in observations.items()}
                    methods={'explicit_only':base.score(explicit),'implicit_only':base.score(weak,route['algorithm']),
                             'mixed':base.score(combined,mixed['algorithm'])}
                    seen=set(observations);negative={r[1] for r in held if not r[2]}
                    for name,values in methods.items():records[name].append(query_metrics(values,base.game_ids,seen,target,negative)[0])
                mixreport[f'{history}:{count}']={'methods':{a:summarize(r) for a,r in records.items()},
                   'paired_mixed_vs_explicit':paired(records['mixed'],records['explicit_only'])}
    report={'protocol':public_summary(protocol),'frozen_config':locked,'explicit':normal,'cold_items':cold,'mixed':mixreport,
       'duration_seconds':time.monotonic()-started,'fresh_user_test_used_for_tuning':False,
       'limitations':['One source/split; bounded1500 historically observed games','No fresh50/100-history users remain after excluding prior exposures',
           'Metadata is static; reviewed-game library proxy and snapshot hours have exposure/temporal bias',
           'Per-cohort intervals use distinct users; cohorts overlap and are not a causal learning curve',
           'Cold results condition on having a cold positive; not a true new-release temporal evaluation']}
    atomic_json(data_dir/'fresh-v2-results.json',report);atomic_json(Path('docs/fresh-v2-results.json'),report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,default=Path('data'));p.add_argument('--content-path',type=Path,required=True)
    a=p.parse_args();r=confirm(a.data_dir,a.content_path);print('Frozen confirmation complete',r['duration_seconds'],flush=True)
