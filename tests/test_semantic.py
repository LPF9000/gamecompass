"""Small numerical fixtures; these are not training/evaluation evidence."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy import sparse

spec=importlib.util.spec_from_file_location('semantic_benchmark',Path(__file__).parents[1]/'scripts'/'benchmark_semantic.py')
semantic=importlib.util.module_from_spec(spec)
spec.loader.exec_module(semantic)


def test_signed_centroid_distinguishes_negative_from_unknown():
    features=np.array([[1.,0.],[0.,1.],[1.,1.]])
    ids={'a':0,'b':1,'c':2}
    scores=semantic.signed_centroid_scores(features,{'a':1,'b':-1},ids)
    assert scores[0] > scores[2] > scores[1]
    assert np.allclose(scores,semantic.signed_centroid_scores(sparse.csr_matrix(features),{'a':1,'b':-1},ids))
    assert np.all(semantic.signed_centroid_scores(features,{'missing':1},ids)==0)


def test_full_candidates_exclude_support_and_need_no_user_embedding():
    f=np.array([[1.,0.],[0.9,0.1],[0.,1.]])
    result=semantic.evaluate_features(f,['support','target','other'],[({'support':1},{'support'},{'target'},set())])
    assert result['users']==1
    assert result['metrics']['recall@10']['mean']==1
    assert result['catalog_coverage@20']==2/3
    assert result['metrics']['ndcg@10']['mean']==1
