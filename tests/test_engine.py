from pathlib import Path
import json
import numpy as np
from scipy import sparse
import pytest
from gamecompass.models import Model
from gamecompass.store import Store
from gamecompass.engine import Recommender
from gamecompass.personal import evaluate_profile, record_outcome
from gamecompass.registry import activate


@pytest.fixture
def ready(tmp_path):
    store=Store(tmp_path)
    games=[{'game_id':f'steam:{i}','title':f'Game {i}', 'description':'exploration story' if i<4 else 'combat racing',
            'genres':['Adventure'] if i<4 else ['Racing'],'platforms':['PC']} for i in range(8)]
    for game in games: store.upsert_game(game)
    model=Model.fit(games,[g['game_id'] for g in games[:6]],sparse.csr_matrix(np.array([[1,1,0,0,0,0],[0,1,1,0,0,0],[0,0,0,1,1,1]],dtype=float)))
    model.selected='itemknn'
    model.save(tmp_path/'models'/'test-v1')
    activate(tmp_path,'test-v1')
    return store, store.create_profile('Tester')


def test_model_recommendations_update_and_filter(ready,tmp_path):
    store,p=ready
    store.set_feedback(p,'steam:0',2,'completed')
    store.set_feedback(p,'steam:4',-2,'abandoned')
    store.set_feedback(p,'steam:2',None,'unplayed',True)
    results=Recommender(tmp_path).recommend(p,limit=8)
    assert not {'steam:0','steam:4','steam:2'} & {r['game_id'] for r in results}
    assert Recommender(tmp_path).recommend(p,platform='PS5')==[]
    assert all('probability' in r['explanation'] for r in results)
    # New profile has no population user embedding and remains independent.
    q=store.create_profile('Independent')
    assert 'not personalized' in Recommender(tmp_path).recommend(q)[0]['explanation']
    assert store.feedback(q)==[]


def test_new_game_support_content_fallback(ready,tmp_path):
    store,p=ready
    store.set_feedback(p,'steam:6',2,'played')
    result=Recommender(tmp_path).recommend(p,limit=1)[0]
    assert 'content fallback' in result['explanation']
    assert result['game_id']=='steam:4' or result['game_id']=='steam:5' or result['game_id']=='steam:7'


def test_model_pointer_traversal_and_corruption(ready,tmp_path):
    with pytest.raises(ValueError): activate(tmp_path,'../../invalid')
    (tmp_path/'models'/'current.json').write_text(json.dumps({'version':'../../private'}))
    with pytest.raises(ValueError): Recommender(tmp_path)._model()


def test_personal_holdout_preserves_feedback_and_deletes_data(ready,tmp_path):
    store,p=ready
    for i in range(6): store.set_feedback(p,f'steam:{i}',2 if i<3 else -1,'played')
    before=store.feedback(p)
    report=evaluate_profile(tmp_path,p)
    assert report['support_count']+report['holdout_count']==6
    assert report['kind']=='predicting_withheld_already_played_preferences'
    assert store.feedback(p)==before
    record_outcome(tmp_path,p,'steam:3','recommendation_relevance',-1)
    payload=store.export_profile(p)
    assert payload['personal_evaluations'] and payload['outcomes']
    store.delete_profile(p)
    with store.connection() as db:
        assert db.execute('SELECT count(*) FROM outcomes').fetchone()[0]==0
        assert db.execute('SELECT count(*) FROM personal_evaluations').fetchone()[0]==0


def test_new_metadata_item_after_training_without_retraining(ready,tmp_path):
    store,p=ready
    store.upsert_game({'game_id':'steam:999','title':'New Release','description':'exploration story','genres':['Adventure'],'platforms':['PC']})
    model=Recommender(tmp_path)._model()
    assert 'steam:999' in model.game_ids and 'steam:999' not in model.cf_game_ids
    store.set_feedback(p,'steam:999',2,'played')
    result=Recommender(tmp_path).recommend(p,limit=1)[0]
    assert 'content fallback' in result['explanation']
    assert result['game_id'] in {'steam:0','steam:1','steam:2','steam:3'}


def test_imported_implicit_personalization_and_explicit_precedence(ready,tmp_path):
    store,p=ready
    manifest_path=tmp_path/'models'/'test-v1'/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest['implicit']={'weighting':'log1p_hours','algorithm':'itemknn','selection_by_history':{}}
    manifest_path.write_text(json.dumps(manifest))
    store.import_library(p,[{'game_id':'steam:0','title':'Game0','playtime_minutes':120}])
    result=Recommender(tmp_path).recommend(p,limit=1)[0]
    assert 'ownership/playtime' in result['explanation']
    assert store.feedback(p)[0]['rating'] is None
    store.set_feedback(p,'steam:0',2,'played')
    explicit=Recommender(tmp_path).recommend(p,limit=1)[0]
    assert 'explicit likes and dislikes' in explicit['explanation']
