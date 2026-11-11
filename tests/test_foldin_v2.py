import inspect
import json
import numpy as np
import pytest
from gamecompass.foldin import FoldInConfig, FoldInModel, load_item_factors
from gamecompass.models import sha256


def test_empty_unknown_and_neutral_use_population_prior():
    model=FoldInModel(['a','b'],np.eye(2),FoldInConfig(),np.array([3.,2.]))
    for support in ({},{'unknown':1},{'a':0}):
        np.testing.assert_array_equal(model.score(support),[3.,2.])
    assert 'user_id' not in inspect.signature(FoldInModel.score).parameters


def test_signed_sum_matches_preexisting_item_aggregation():
    factors=np.array([[1.,2.],[3.,-1.],[0.,2.]])
    model=FoldInModel(['a','b','c'],factors,FoldInConfig(method='sum',normalization='none'))
    np.testing.assert_allclose(model.score({'a':2,'b':-1}),factors@(2*factors[0]-factors[1]))
    np.testing.assert_array_equal(factors,[[1.,2.],[3.,-1.],[0.,2.]])


def test_ridge_uses_only_observed_labels_and_confidence():
    model=FoldInModel(['a','b','c'],np.eye(3),FoldInConfig(method='ridge',normalization='none',regularization=1))
    np.testing.assert_allclose(model.score({'a':1,'b':-2}),[.5,-2/3,0])
    extended=FoldInModel(['a','b','c','unobserved'],np.vstack([np.eye(3),[100.,100.,100.]]),model.config)
    np.testing.assert_allclose(extended.score({'a':1,'b':-2})[:3],model.score({'a':1,'b':-2}))


def test_normalized_centroids_resist_item_norm_and_class_count():
    factors=np.array([[1.,0.],[10.,0.],[0.,2.],[0.,1.]])
    model=FoldInModel(['a','b','c','d'],factors,FoldInConfig(method='centroid',normalization='l2'))
    np.testing.assert_allclose(model.user_vector({'a':1,'b':1,'c':-1}),[1.,-1.])
    assert model.score({'a':1,'b':1,'c':-1})[0]>model.score({'a':1,'b':1,'c':-1})[2]
    np.testing.assert_allclose(model.user_vector({'a':.1,'c':-2}),[.1,-2.])


def test_logistic_observed_feedback_direction_and_unknown_invariance():
    config=FoldInConfig(method='logistic',normalization='none',regularization=.1)
    model=FoldInModel(['a','b'],np.eye(2),config)
    scores=model.score({'a':1,'b':-1})
    assert scores[0]>0>scores[1]
    extended=FoldInModel(['a','b','c'],np.vstack([np.eye(2),[99,99]]),config)
    np.testing.assert_allclose(extended.score({'a':1,'b':-1})[:2],scores)
    assert model.logistic_nonconverged==0


def test_invalid_config_arrays_and_feedback():
    with pytest.raises(ValueError): FoldInConfig(regularization=0)
    with pytest.raises(ValueError): FoldInConfig(normalization='unknown')
    with pytest.raises(ValueError): FoldInModel(['a','a'],np.eye(2),FoldInConfig())
    with pytest.raises(ValueError): FoldInModel(['a'],np.array([[np.nan]]),FoldInConfig())
    model=FoldInModel(['a'],np.zeros((1,2)),FoldInConfig(normalization='center_l2'))
    np.testing.assert_array_equal(model.score({'a':1}),[0])
    with pytest.raises(ValueError): model.score({'a':float('nan')})


def test_loader_rejects_user_arrays_without_reading_them_and_checks_integrity(tmp_path):
    path=tmp_path/'factors.npz'
    np.savez(path,game_ids=np.array(['a']),factors=np.array([[1.]]),user_factors=np.array([object()],dtype=object))
    (tmp_path/'manifest.json').write_text(json.dumps({'factor_sha256':sha256(path)}))
    with pytest.raises(ValueError,match='item-only'): load_item_factors(tmp_path)
    np.savez(path,game_ids=np.array(['a']),factors=np.array([[1.]]))
    (tmp_path/'manifest.json').write_text(json.dumps({'factor_sha256':sha256(path)}))
    ids,factors,_=load_item_factors(tmp_path)
    assert ids==['a'] and factors.shape==(1,1)
    path.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='checksum'): load_item_factors(tmp_path)


def test_ridge_primal_branch_with_more_observations_than_features():
    factors=np.array([[1.,0.],[0.,1.],[1.,0.],[0.,1.],[1.,1.]])
    model=FoldInModel(['a','b','c','d','unobserved'],factors,FoldInConfig(method='ridge',normalization='none',regularization=1))
    np.testing.assert_allclose(model.score({'a':1,'b':1,'c':1,'d':-1}),[2/3,0,2/3,0,2/3])
