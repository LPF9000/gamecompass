"""Tests use small synthetic fixtures, never presented as benchmark data."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from scipy import sparse

spec = importlib.util.spec_from_file_location('content_projection', Path(__file__).resolve().parents[1] / 'scripts' / 'benchmark_content_projection.py')
projection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(projection)


def test_projection_generalizes_known_linear_features_without_cold_targets():
    warm_features = sparse.eye(3, format='csr')
    warm_factors = np.array([[1.,0.],[0.,1.],[1.,1.]])
    cold_features = sparse.csr_matrix([[1.,1.,0.],[0.,0.,0.]])
    result = projection.fit_projection(warm_features,warm_factors,cold_features,alpha=1)
    assert np.allclose(result[0],np.array([1.,1.])/np.sqrt(2))
    assert np.allclose(result[1],0)
    with pytest.raises(ValueError):
        projection.fit_projection(warm_features,warm_factors,cold_features,alpha=0)


def test_signed_fold_in_cannot_consume_cold_item_vectors():
    factors=np.eye(2,dtype=np.float32)
    result=projection.user_factor({'warm_a':1,'warm_b':-1,'cold_withheld':1000},{'warm_a':0,'warm_b':1},factors)
    assert np.allclose(result,np.array([1.,-1.])/np.sqrt(2))
    assert np.allclose(projection.user_factor({'cold_withheld':1},{'warm_a':0,'warm_b':1},factors),0)
