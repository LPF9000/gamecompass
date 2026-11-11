import numpy as np
from scipy import sparse
import pytest
from gamecompass.models import Model, ALGORITHMS


def small_model():
    games = [{"game_id": f"g{i}", "description": "exploration story" if i < 2 else "racing speed"} for i in range(4)]
    x = sparse.csr_matrix([[1,1,0],[1,1,0],[0,1,1],[0,0,1]])
    return Model.fit(games, ["g0", "g1", "g2"], x)


def test_signed_feedback_and_cold_items():
    model = small_model()
    for algorithm in ALGORITHMS:
        assert model.score({"never-a-training-user": 1}, algorithm).shape == (4,)
        assert np.isfinite(model.score({"g0": 2, "g2": -2}, algorithm)).all()
    assert model.score({"g2":1}, "content")[3] > model.score({"g0":1}, "content")[3]
    assert np.allclose(model.score({"g0":-1}, "itemknn"), -model.score({"g0":1}, "itemknn"))


def test_roundtrip_and_corruption(tmp_path):
    model = small_model()
    path = tmp_path / "m"
    model.save(path)
    loaded = Model.load(path)
    for algorithm in ALGORITHMS:
        assert np.allclose(model.score({"g0":1},algorithm),loaded.score({"g0":1},algorithm))
    with (path / "weights.npz").open("ab") as f:
        f.write(b"corrupted")
    with pytest.raises(ValueError,match="integrity"):
        Model.load(path)


def test_duplicate_catalog_and_unknown_algorithm():
    with pytest.raises(ValueError,match="Duplicate"):
        Model.fit([{"game_id":"same"},{"game_id":"same"}], ["same"], sparse.csr_matrix([[1]]))
    with pytest.raises(ValueError,match="Unknown"):
        small_model().score({}, "fictional")


def test_portable_new_content_features():
    model = small_model()
    assert model.content_features([{"description":"racing speed"}]).shape == (1,4096)


def test_validated_history_route_and_hybrid_prior_roundtrip(tmp_path):
    model = small_model()
    model.selection_by_history = {'0':'popularity','3':'ease'}
    assert model.algorithm_for({'g0':1}) == 'popularity'
    assert model.algorithm_for({'g0':1,'g1':1,'g2':-1}) == 'ease'
    model.hybrid_popularity = .25
    model.hybrid_alpha = .1
    path = tmp_path / 'routed'
    model.save(path)
    loaded = Model.load(path)
    assert loaded.selection_by_history == model.selection_by_history
    assert loaded.hybrid_popularity == .25
    support={'g0':1,'g1':1,'g2':-1}
    assert np.allclose(loaded.score(support), model.score(support,'ease'))
