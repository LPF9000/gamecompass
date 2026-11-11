import json

import pytest

from gamecompass.evaluation import HISTORY_LENGTHS, cold_item_queries, cold_queries
from gamecompass.protocol import (cohort_users, create_protocol, freeze_experiment_config, legacy_exposed_user_ids,
                                 public_summary, select_cohort)


def rows(user, n):
    return [(user, f"game:{i}", int(i % 3 != 1), i) for i in range(n)]


def test_exposure_reconstructs_all_history_query_eligibilities():
    users = {f"u{i:03}": rows(f"u{i:03}", i + 1) for i in range(120)}
    exposed = legacy_exposed_user_ids(users, [f"game:{i}" for i in range(121)], max_users=3)
    expected = set()
    # Single-user query calls preserve exact existing eligibility semantics.
    for history in HISTORY_LENGTHS:
        expected.update([user for user in sorted(users) if cold_queries({user: users[user]}, history)][:3])
    from gamecompass.evaluation import cold_item_partition
    cold = {f"game:{i}" for i in range(121) if cold_item_partition(f"game:{i}")}
    for history in (1, 3, 5):
        expected.update([user for user in sorted(users) if cold_item_queries({user: users[user]}, cold, history)][:3])
    assert exposed == expected
    assert "u000" in exposed  # zero-history published results also expose users
    assert "u119" not in exposed
    assert "u119" in legacy_exposed_user_ids(users, [], max_users=3, additional_ids=["u119"])


def test_fresh_cohort_order_is_independent_of_dict_order_and_excludes_whole_users():
    users = {f"u{i:03}": rows(f"u{i:03}", 25) for i in range(100)}
    exposed = {"u001", "u002", "u003"}
    selected = select_cohort(users, exposed, 3, cap=20)
    assert selected == select_cohort(dict(reversed(list(users.items()))), exposed, 3, cap=20)
    assert len(selected) == 20 and not set(selected) & exposed
    assert selected != sorted(selected)
    assert select_cohort(users, exposed, 50) == []
    assert select_cohort(users, exposed, 3, cap=20, seed=1) != selected


def test_config_freeze_is_immutable_and_public_summary_has_no_ids(tmp_path):
    manifest = {key: "value" for key in ("protocol_version", "seed", "cap_per_history", "base_split_seed", "dataset_sha256", "base_version", "base_split_sha256", "base_weight_checksums", "cf_ids_sha256", "legacy_primary_cap", "legacy_temporal_overlap_count", "original_test_users", "remaining_unexposed_test_users", "available_by_history", "selection", "candidate_strategy", "model_selection", "scope")}
    manifest.update({"legacy_exposed_user_ids": ["private-user-id"], "cohorts": {"1": ["fresh-private-user-id"]}})
    summary = public_summary(manifest)
    assert "private-user-id" not in json.dumps(summary)
    assert summary["selected_by_history"] == {"1": 1}
    first = freeze_experiment_config(tmp_path, "test", {"alpha": .5}, manifest)
    assert freeze_experiment_config(tmp_path, "test", {"alpha": .5}, manifest) == first
    with pytest.raises(ValueError, match="Immutable"):
        freeze_experiment_config(tmp_path, "test", {"alpha": .1}, manifest)


def test_cohort_guards_universe_and_exposure():
    import hashlib
    from gamecompass.protocol import _canonical
    prepared = {"cf_ids": ["game:0", "game:1"], "test_users": {"u": rows("u", 3)}}
    manifest = {"cf_ids_sha256": hashlib.sha256(_canonical(prepared["cf_ids"])).hexdigest(),
                "legacy_exposed_user_ids": [], "cohorts": {"1": ["u"]}}
    assert list(cohort_users(manifest, prepared, 1)) == ["u"]
    manifest["legacy_exposed_user_ids"] = ["u"]
    with pytest.raises(ValueError, match="exposure"):
        cohort_users(manifest, prepared, 1)
    manifest["legacy_exposed_user_ids"] = []
    prepared["cf_ids"] = ["different"]
    with pytest.raises(ValueError, match="universe"):
        cohort_users(manifest, prepared, 1)


def test_protocol_creation_freezes_provenance_and_rejects_changes(tmp_path):
    from gamecompass.models import sha256
    data = tmp_path
    (data / "population.sqlite").write_bytes(b"fixed dataset bytes")
    artifact = data / "models" / "base"
    artifact.mkdir(parents=True)
    split_path = data / "experiments" / "base" / "split.json"
    split_path.parent.mkdir(parents=True)
    split_path.write_text(json.dumps({"seed": 42, "cf_ids": ["game:0", "game:1"],
        "test_users": {f"u{i:04}": rows(f"u{i:04}", 3) for i in range(400)}}))
    for name in ("weights.npz", "knn.npz", "content.npz"):
        (artifact / name).write_bytes(name.encode())
    model = {"source_sha256": sha256(data / "population.sqlite"), "split_sha256": sha256(split_path),
             "checksums": {name: sha256(artifact / name) for name in ("weights.npz", "knn.npz", "content.npz")}}
    (artifact / "manifest.json").write_text(json.dumps(model))
    manifest = create_protocol(data, base_version="base", cap=20)
    assert len(manifest["cohorts"]["1"]) == 20
    assert manifest["remaining_unexposed_test_users"] == 100
    assert create_protocol(data, base_version="base", cap=20) == manifest
    before = (data / "protocols" / "fresh-users-v2.json").read_bytes()
    with pytest.raises(ValueError, match="Immutable"):
        create_protocol(data, base_version="base", cap=10)
    assert (data / "protocols" / "fresh-users-v2.json").read_bytes() == before
    (data / "population.sqlite").write_bytes(b"changed dataset")
    with pytest.raises(ValueError, match="differs"):
        create_protocol(data, base_version="base", cap=20)
