import numpy as np
import pytest
from gamecompass.blending import CandidateGenerator,ScoreBlend,normalized_channels,fit_blend


def test_candidate_union_filters_seen_and_keeps_channel_specialists():
    channels={'a':np.array([1,0,.5,0]),'b':np.array([0,1,.5,0])}
    pool=CandidateGenerator(1).generate(['a','b','c','d'],channels,{'a'})
    assert set(pool)=={1,2}


def test_blend_roundtrip_and_finite_validation(tmp_path):
    model=ScoreBlend(['baseline','tags'],np.array([.75,.25]),{})
    channels={'baseline':np.array([0,2]),'tags':np.array([2,0])}
    assert np.allclose(model.score(channels),[.25,.75])
    path=tmp_path/'blend.json';model.save(path)
    assert np.allclose(ScoreBlend.load(path).score(channels),model.score(channels))
    with pytest.raises(ValueError):normalized_channels({'a':np.array([np.nan])},['a'])
    with pytest.raises(ValueError):ScoreBlend(['a'],np.array([-1]),{})


def test_selection_uses_only_supplied_validation_targets():
    names=['baseline','tags'];features=np.array([[[1,0],[0,1],[0,.5]]],dtype=np.float32)
    target=np.array([[0,1,0]],dtype=bool);allowed=np.ones((1,3),dtype=bool)
    model,trials=fit_blend(names,features,target,allowed,['a','b','c'])
    assert model.weights[1] > 0
    assert trials and model.metadata['validation_ndcg@20']==1
