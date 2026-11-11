import json
import sqlite3

import pytest

from gamecompass.store import Store


@pytest.fixture
def store(tmp_path):
    result = Store(tmp_path)
    result.upsert_game({"game_id": "steam:1", "title": "Example", "platforms": ["Windows"]})
    return result


def test_profiles_persist_and_isolate(store):
    a = store.create_profile("Alice")
    b = store.create_profile("Bob")
    store.set_feedback(a, "steam:1", 2, "abandoned", reasons=["Pacing"])
    assert not store.feedback(b)
    row = Store(store.data_dir).feedback(a)[0]
    assert row["rating"] == 2 and row["status"] == "abandoned"
    store.delete_profile(a)
    assert [row["id"] for row in store.profiles()] == [b]
    with store.connection() as db:
        assert db.execute("SELECT count(*) FROM feedback").fetchone()[0] == 0


def test_feedback_validation_and_update(store):
    profile = store.create_profile("Person")
    store.set_feedback(profile, "steam:1", -1, "played", True)
    store.set_feedback(profile, "steam:1", 0, "wishlist")
    assert len(store.feedback(profile)) == 1
    assert store.feedback(profile)[0]["rating"] == 0
    for rating in (3, True, 1.5, "1"):
        with pytest.raises(ValueError):
            store.set_feedback(profile, "steam:1", rating)
    with pytest.raises(ValueError):
        store.set_feedback(profile, "missing", 2)
    with pytest.raises(ValueError):
        store.set_feedback(profile, "steam:1", 1, "invalid")
    with pytest.raises(ValueError):
        store.set_feedback("missing", "steam:1", 1)


def test_import_preserves_explicit_negative(store):
    profile = store.create_profile("Player")
    store.set_feedback(profile, "steam:1", -2, "abandoned")
    store.import_library(profile, [{"game_id": "steam:1", "title": "Example", "playtime_minutes": 900}])
    row = store.feedback(profile)[0]
    assert row["rating"] == -2 and row["status"] == "abandoned"
    assert store.implicit(profile)[0]["playtime_minutes"] == 900
    assert store.profile_connections(profile)[0]["game_count"] == 1


def test_export_import_roundtrip_and_atomic_rejection(store):
    profile = store.create_profile("Person")
    store.set_feedback(profile, "steam:1", 1, "completed", reasons=["Story"])
    store.import_library(profile, [{"game_id": "steam:1", "title": "Example", "playtime_minutes": 10}])
    exported = store.export_profile(profile)
    assert "STEAM_API_KEY" not in json.dumps(exported)
    clone = store.import_profile(exported)
    assert store.feedback(clone)[0]["reasons"] == ["Story"]
    assert store.implicit(clone)[0]["playtime_minutes"] == 10
    before = len(store.profiles())
    exported["feedback"][0]["rating"] = 9
    with pytest.raises(ValueError):
        store.import_profile(exported)
    assert len(store.profiles()) == before


def test_catalog_sync_and_literal_search(tmp_path):
    source = Store(tmp_path / "source")
    source.upsert_game({"game_id": "steam:42", "title": "100% Example", "metadata": {"alternative_titles": ["Old Name"]}})
    import shutil
    shutil.copy(source.path, tmp_path / "population.sqlite")
    target = Store(tmp_path)
    assert target.game("steam:42")["title"] == "100% Example"
    assert len(target.search("%")) == 1
    assert target.search("Old Name")[0]["game_id"] == "steam:42"
    assert not target.search("' OR 1=1 --")
    assert target.catalog_count() == 1


def test_incompatible_database_rejected(tmp_path):
    with sqlite3.connect(tmp_path / "local.sqlite") as db:
        db.execute("PRAGMA user_version=999")
    with pytest.raises(ValueError, match="newer"):
        Store(tmp_path)


def test_failed_library_import_does_not_mutate(store):
    profile = store.create_profile("Player")
    with pytest.raises(ValueError):
        store.import_library(profile, [{"game_id": "steam:2", "title": "New"}, {"game_id": "steam:3", "title": "Bad", "playtime_minutes": -1}])
    assert not store.game("steam:2")
    assert not store.feedback(profile)


def test_sync_preserves_enrichment_and_skips_unchanged(tmp_path):
    source = Store(tmp_path / "source")
    source.upsert_game({"game_id": "steam:42", "title": "Population title"})
    import shutil
    shutil.copy(source.path, tmp_path / "population.sqlite")
    target = Store(tmp_path)
    target.upsert_game({"game_id": "steam:42", "title": "Updated title", "description": "Enriched locally"})
    assert target.sync_catalog() == 0
    assert target.game("steam:42")["title"] == "Updated title"
    assert target.game("steam:42")["description"] == "Enriched locally"


def test_rejects_duplicate_feedback_and_invalid_collections(store):
    profile = store.create_profile("Player")
    store.set_feedback(profile, "steam:1", 1)
    payload = store.export_profile(profile)
    payload["feedback"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        store.import_profile(payload)
    with pytest.raises(ValueError):
        store.import_profile({"schema_version": 1, "profile": []})
    with pytest.raises(ValueError):
        store.set_feedback(profile, "steam:1", 1, reasons="")


def test_export_import_and_delete_include_personal_evaluations(store):
    from gamecompass.personal import record_outcome
    profile = store.create_profile("Player")
    record_outcome(store.data_dir, profile, "steam:1", "recommendation_relevance", 1)
    with store.connection() as db:
        db.execute("INSERT INTO personal_evaluations VALUES (?,?,?)", (profile, '{"note":"small diagnostic"}', 1234))
    exported = store.export_profile(profile)
    assert len(exported["games"]) == 1
    clone = store.import_profile(exported)
    assert store.personal_history(clone)["outcomes"][0]["value"] == 1
    store.delete_profile(profile)
    with store.connection() as db:
        assert db.execute("SELECT count(*) FROM outcomes WHERE profile_id=?", (profile,)).fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM personal_evaluations WHERE profile_id=?", (profile,)).fetchone()[0] == 0


def test_manual_csv_import_preserves_separate_playtime(tmp_path):
    from gamecompass.ui import import_csv
    store=Store(tmp_path);p=store.create_profile('CSV')
    import_csv(store,p,'game_id,title,rating,status,playtime_minutes\nsteam:100,Test Game,,played,120\n')
    assert store.feedback(p)[0]['rating'] is None
    assert store.implicit(p)[0]['playtime_minutes']==120
    assert len(store.profiles())==1
