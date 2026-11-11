import json

import numpy as np
import pytest

from gamecompass.quality import QualityPrior


def test_approval_prior_separates_count_from_opinion():
    rows = [("a", "popular", 1, 0)] * 8 + [("b", "popular", 0, 0)] * 2
    rows += [("c", "niche", 1, 0)] * 2 + [("d", "bad", 0, 0)] * 2
    prior = QualityPrior.fit(rows, ["popular", "niche", "bad", "empty"])
    assert prior.global_rate == pytest.approx(10 / 14)
    channels = prior.channels(["popular", "niche", "bad", "empty", "unknown"], 5)
    assert channels["count"].tolist() == [10, 2, 2, 0, 0]
    assert prior.score(["popular"], 5)[0] == pytest.approx((8 + 5 * prior.global_rate) / 15)
    assert prior.score(["unknown"], 5)[0] == prior.global_rate
    assert channels["centered_log_odds"][-1] == pytest.approx(0)
    assert channels["logcount"][0] > channels["logcount"][1]
    assert prior.score(["bad"], 100)[0] > prior.score(["bad"], 5)[0]


def test_safe_artifact_has_only_aggregate_counts(tmp_path):
    prior = QualityPrior.fit([("PRIVATE_USER_1", "steam:1", 1, 10), ("PRIVATE_USER_2", "steam:2", 0, 20)], ["steam:1", "steam:2"])
    prior.save(tmp_path)
    restored = QualityPrior.load(tmp_path)
    np.testing.assert_array_equal(restored.score(["steam:1", "unknown"]), prior.score(["steam:1", "unknown"]))
    assert "PRIVATE_USER" not in (tmp_path / "manifest.json").read_text()
    (tmp_path / "counts.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        QualityPrior.load(tmp_path)


def test_quality_rejects_bad_labels_strength_and_ids():
    with pytest.raises(ValueError, match="Duplicate"):
        QualityPrior.fit([], ["a", "a"])
    with pytest.raises(ValueError, match="binary"):
        QualityPrior.fit([("u", "a", -1, 0)], ["a"])
    prior = QualityPrior.fit([], ["a"])
    assert prior.score(["a", "missing"]).tolist() == [0.5, 0.5]
    for strength in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="strength"):
            prior.score(["a"], strength)
    assert np.isfinite(QualityPrior.fit([("u", "a", 1, 0)], ["a"]).channels(["a"])["centered_log_odds"]).all()


def test_quality_training_scope_excludes_unprovided_targets():
    # Training-only input excludes validation/test and warm rows by construction.
    prior = QualityPrior.fit([("train", "a", 1, 0), ("train", "outside", 0, 0)], ["a", "b"])
    assert prior.positive.tolist() == [1, 0]
    assert prior.negative.tolist() == [0, 0]
    assert prior.metadata["outside_scope_rows"] == 1
    assert prior.channels(["b"])["count"].tolist() == [0]
