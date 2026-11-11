"""Synthetic fixtures validate content serialization/semantics, not model accuracy."""
import json
import numpy as np
import pytest
from gamecompass.content import ContentFeatures, ContentModel, atomic_tags, fit_ridge_projection


def small_model():
    games = [
        {'game_id':'a','description':'explore ancient ruins','metadata':json.dumps({'tags':['Open World','Story Rich']})},
        {'game_id':'b','description':'drive racing cars','metadata':{'tags':['Racing','Driving']}},
        {'game_id':'c','description':'explore forgotten ruins','metadata':{'tags':['Open World','Exploration']}},
    ]
    model=ContentModel()
    model.features=ContentFeatures.fit(games,min_df=1,max_description_features=100)
    model.tags,model.words=model.features.transform(games)
    model.game_ids=['a','b','c'];model.warm_ids=['a','b'];model.warm_factors=np.eye(2,dtype=np.float32)
    model.cosine_tag_weight=.75;model.projection_tag_weight=.5
    matrix=ContentFeatures.combine(model.tags,model.words,.5)
    model.projection=fit_ridge_projection(matrix[:2],model.warm_factors,1)
    model._indexes()
    return model


def test_real_metadata_tags_are_atomic_and_distinct_from_description_words():
    game={'description':'race through a city','metadata':'{"tags":["Open World","Story Rich"," Open   World "]}', 'genres':'["STORY RICH"]'}
    features=ContentFeatures.fit([game,{'description':'race through a town'}],min_df=1)
    assert atomic_tags(game)==['open world','story rich']
    assert set(features.tag_vocabulary)=={'open world','story rich'}
    assert 'open world' not in features.description_vocabulary
    tags,words=features.transform([game])
    assert np.linalg.norm(tags.toarray())==pytest.approx(1)
    assert np.linalg.norm(words.toarray())==pytest.approx(1)


def test_portable_features_and_projection_roundtrip(tmp_path):
    model=small_model();path=tmp_path/'artifact';model.save(path)
    loaded=ContentModel.load(path)
    for algorithm in ('tags','description','cosine','projection'):
        expected=model.score({'a':1,'b':-1},algorithm=algorithm)
        assert np.allclose(loaded.score({'a':1,'b':-1},algorithm=algorithm),expected)
        assert np.allclose(loaded.score({'a':1,'b':-1},['b','a'],algorithm=algorithm),expected[[1,0]])
    with np.load(path/'weights.npz',allow_pickle=False) as arrays:
        assert all(arrays[k].dtype.kind!='O' for k in arrays.files)
    with (path/'weights.npz').open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='integrity'):ContentModel.load(path)


def test_new_game_metadata_and_signed_new_support_without_population_fit():
    model=small_model()
    fresh={'game_id':'fresh','description':'','metadata':{'tags':['Open World']}}
    for algorithm in ('tags','cosine','projection'):
        scores=model.score({'a':1},[fresh,{'game_id':'missing'}],algorithm=algorithm)
        assert scores[0]>scores[1]
        cold=model.score({'fresh':1},['a','b'],support_games=[fresh],algorithm=algorithm)
        assert cold[0]>cold[1]
        assert np.allclose(model.score({'fresh':-1},['a','b'],support_games=[fresh],algorithm=algorithm),-cold)
    assert np.allclose(model.score({'unknown':1}),0)


def test_unknown_and_missing_fields_give_zero_evidence_and_cache_known_candidates():
    model=small_model()
    tags,words=model.features.transform([{'metadata':'broken','description':''},{'metadata':{'tags':['never observed tag']}}])
    assert tags.nnz==words.nnz==0
    first=model.score({'a':1},['a','c'],algorithm='tags')
    count=len(model._candidate_cache)
    second=model.score({'a':1},['a','c'],algorithm='tags')
    assert count==len(model._candidate_cache)
    assert np.allclose(first,second)
    with pytest.raises(ValueError):model.score({},algorithm='made-up')
    with pytest.raises(ValueError):ContentFeatures.combine(tags,words,1.1)


def test_one_game_catalog_and_mixed_new_cached_candidate_alignment():
    features=ContentFeatures.fit([{'description':'explore ancient ruins','metadata':{'tags':['Open World']}}])
    assert features.description_vocabulary
    model=small_model()
    fresh={'game_id':'new','metadata':{'tags':['Open World']},'description':''}
    for algorithm in ('tags','description','cosine','projection'):
        mixed=model.score({'a':1},['b',fresh,'a'],algorithm=algorithm)
        assert mixed[0]==pytest.approx(model.score({'a':1},['b'],algorithm=algorithm)[0])
        assert mixed[1]==pytest.approx(model.score({'a':1},[fresh],algorithm=algorithm)[0])
        assert mixed[2]==pytest.approx(model.score({'a':1},['a'],algorithm=algorithm)[0])
