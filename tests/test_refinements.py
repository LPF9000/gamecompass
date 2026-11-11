from datetime import datetime, timezone

import pytest

from gamecompass.refinements import (
    RecommendationFilters, available_genres, available_platforms, genre_facts,
    release_facts,
)


def game(**kwargs):
    return {"game_id": "steam:123", "title": "Example HD", "platforms": ["Windows"],
            "genres": [], "metadata": {}, **kwargs}


def test_kaggle_listing_date_is_never_original_date():
    value = game(metadata={"source": "kaggle-steam-v28", "release_date": "2013-12-05"})
    facts = release_facts(value)
    assert facts["edition_year"] == 2013
    assert facts["original_year"] is None
    assert facts["release_kind"] == "unknown"  # HD in a title is not evidence.
    assert not RecommendationFilters(year_min=2010, year_max=2019).matches(value)
    assert RecommendationFilters(year_min=2010, year_max=2019, era_basis="edition").matches(value)
    assert not RecommendationFilters(edition_policy="remakes_remasters_only").matches(value)


def test_manual_original_and_remaster_have_independent_eras():
    value = game(metadata={"manual_release": {"source": "user-provided", "original_date": "1997-01-31",
                                              "edition_date": "2013-12-05", "release_kind": "remaster"}})
    facts = release_facts(value)
    assert facts["original_year"] == 1997 and facts["edition_year"] == 2013
    assert facts["original_source"] == "user-provided"
    assert RecommendationFilters(year_min=1990, year_max=1999).matches(value)
    assert not RecommendationFilters(year_min=1990, year_max=1999, era_basis="edition").matches(value)
    assert RecommendationFilters(edition_policy="remakes_remasters_only").matches(value)
    assert not RecommendationFilters(edition_policy="originals_only").matches(value)


def test_unverified_original_assertion_and_invalid_dates_ignored():
    value = game(metadata={"original_release_date": "1997", "release_date": "invalid",
                          "manual_release": {"source": "unknown-provider", "original_date": "1997"}})
    assert release_facts(value)["original_year"] is None
    assert release_facts(value)["edition_year"] is None
    assert RecommendationFilters().matches(value)
    assert not RecommendationFilters(year_max=1999).matches(value)


def test_igdb_exact_mapping_and_relationships_are_required():
    stamp = datetime(1997, 1, 31, tzinfo=timezone.utc).timestamp()
    original = {"id": 9, "first_release_date": stamp, "game_type": {"type": "Main Game"}, "remasters": [42]}
    metadata = {"igdb": original, "igdb_mapping": {"status": "matched", "confidence": 1.0, "igdb_id": 9}}
    value = game(metadata=metadata)
    facts = release_facts(value)
    assert facts["original_year"] == 1997
    assert facts["release_kind"] == "original"  # Having remasters does not make the original a remaster.
    original["version_parent"] = 8
    assert release_facts(value)["original_year"] is None
    original.pop("version_parent")
    original["game_type"] = {"type": "Remake"}
    assert release_facts(value)["release_kind"] == "remake"
    assert release_facts(value)["original_year"] is None
    metadata["igdb_mapping"]["status"] = "review"
    assert release_facts(value)["edition_year"] is None


def test_genre_tag_fallback_is_limited_and_labeled():
    value = game(genres=["Adventure"], metadata={"tags": ["RPG", "Story Rich", "Singleplayer", "Adventure", "Anime"]})
    facts = {row["genre"]: row["source"] for row in genre_facts(value)}
    assert facts == {"Adventure": "catalog", "RPG": "catalog tag (inferred)"}
    assert RecommendationFilters(genre="rpg").matches(value)
    assert not RecommendationFilters(genre="Story Rich").matches(value)
    assert available_genres([value]) == ["Adventure", "RPG"]


def test_platforms_are_verified_not_inferred_from_tags_or_other_generations():
    value = game(platforms=["PS5"], metadata={"tags": ["Controller", "Windows", "PlayStation 4"]})
    assert RecommendationFilters(platform="PlayStation 5").matches(value)
    assert not RecommendationFilters(platform="PS4").matches(value)
    assert not RecommendationFilters(platform="Windows").matches(value)
    assert available_platforms([value]) == ["PlayStation 5"]
    assert not RecommendationFilters(platform="PS5").matches(game(platforms=[]))


@pytest.mark.parametrize("arguments", [
    {"year_min": 2010, "year_max": 1990}, {"year_min": True}, {"year_max": 3000},
    {"era_basis": "listing"}, {"edition_policy": "hd-in-title"}, {"genre": ""}, {"platform": 5},
])
def test_invalid_filter_configuration_rejected(arguments):
    with pytest.raises(ValueError):
        RecommendationFilters(**arguments)


def test_discover_refinements_do_not_restrict_search_or_ratings(tmp_path, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from gamecompass.engine import Recommender
    from gamecompass.store import Store
    monkeypatch.setenv("GAMECOMPASS_DATA", str(tmp_path))
    store = Store(tmp_path)
    profile = store.create_profile("Tester")
    store.upsert_game(game(title="Recent Adventure", genres=["Adventure"], metadata={"release_date": "2023-01-01"}))
    calls = []
    def recommend(self, profile_id, **kwargs):
        calls.append(kwargs)
        return []
    monkeypatch.setattr(Recommender, "recommend", recommend)
    ui = Path(__file__).parents[1] / "src" / "gamecompass" / "ui.py"
    app = AppTest.from_file(str(ui), default_timeout=15).run()
    app.radio[0].set_value("Discover").run()
    next(box for box in app.selectbox if box.label == "Era").set_value("1990s").run()
    assert calls[-1]["filters"].year_min == 1990
    assert calls[-1]["filters"].era_basis == "edition"
    next(box for box in app.checkbox if box.label.startswith('Use the original work')).set_value(True).run()
    assert calls[-1]['filters'].era_basis == 'original'
    assert not calls[-1]["filters"].matches(store.game("steam:123"))
    app.radio[0].set_value("Search").run()
    app.text_input[0].set_value("Recent Adventure").run()
    assert not app.exception
    next(box for box in app.selectbox if box.label == "Your enjoyment").set_value("Loved it")
    next(button for button in app.button if button.label == "Save feedback").click().run()
    assert not app.exception
    assert store.feedback(profile)[0]["rating"] == 2


def test_if_you_liked_seed_uses_same_filters_without_changing_ratings(tmp_path, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from gamecompass.engine import Recommender
    from gamecompass.store import Store
    monkeypatch.setenv("GAMECOMPASS_DATA", str(tmp_path))
    store = Store(tmp_path)
    profile = store.create_profile("Tester")
    store.upsert_game(game())
    store.upsert_game(game(game_id="steam:456", title="Candidate"))
    monkeypatch.setattr(Recommender, "recommend", lambda self, profile_id, **kwargs: [])
    calls = []
    def similar(self, game_id, **kwargs):
        calls.append((game_id, kwargs))
        return [dict(store.game("steam:456"), explanation="Mocked engine output for isolated UI test")]
    monkeypatch.setattr(Recommender, "similar", similar, raising=False)
    ui = Path(__file__).parents[1] / "src" / "gamecompass" / "ui.py"
    app = AppTest.from_file(str(ui), default_timeout=15).run()
    app.radio[0].set_value("Search").run()
    app.text_input[0].set_value("Example HD").run()
    next(button for button in app.button if button.label == "If you liked this game").click().run()
    assert not app.exception
    assert calls[-1][0] == "steam:123"
    assert calls[-1][1]["profile_id"] == profile
    assert isinstance(calls[-1][1]["filters"], RecommendationFilters)
    assert not store.feedback(profile)
    assert any(header.value.endswith("Candidate") for header in app.subheader)


def test_manual_edition_dates_are_labeled_user_provided(tmp_path, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from gamecompass.engine import Recommender
    from gamecompass.store import Store
    monkeypatch.setenv("GAMECOMPASS_DATA", str(tmp_path))
    store = Store(tmp_path)
    store.create_profile("Tester")
    monkeypatch.setattr(Recommender, "recommend", lambda self, profile_id, **kwargs: [])
    ui = Path(__file__).parents[1] / "src" / "gamecompass" / "ui.py"
    app = AppTest.from_file(str(ui), default_timeout=15).run()
    app.radio[0].set_value("Settings").run()
    inputs = {field.label: field for field in app.text_input}
    inputs["Canonical game ID"].set_value("steam:42")
    inputs["Game title"].set_value("Known Remaster")
    inputs["Original work release date (optional)"].set_value("1997")
    inputs["This edition release date (optional)"].set_value("2013-12-05")
    next(box for box in app.selectbox if box.label == "Known release type").set_value("remaster")
    next(button for button in app.button if button.label == "Save catalog game").click().run()
    assert not app.exception
    facts = release_facts(store.game("steam:42"))
    assert facts["original_year"] == 1997 and facts["edition_year"] == 2013
    assert facts["original_source"] == "user-provided"
    assert facts["release_kind"] == "remaster"


def test_wikidata_entity_date_never_becomes_original_or_unqualified_console_port():
    game={'game_id':'wikidata:Q123','title':'Example','platforms':['Windows','Nintendo Switch'], 'genres':['RPG'],
          'metadata':{'entity_release_date':'2017-02','wikidata':{'entity_id':'Q123',
            'platform_facts':[{'value':'urn:windows','label':'Microsoft Windows'},{'value':'urn:switch','label':'Nintendo Switch'}],
            'publication_dates':[{'iso':'2024-03','qualifiers':{'P400':['urn:switch']}}]}}}
    assert release_facts(game)['original_year'] is None
    assert release_facts(game)['edition_year']==2017
    assert release_facts(game,platform='Nintendo Switch')['edition_year']==2024
    assert release_facts(game,platform='Windows')['edition_year'] is None
    assert RecommendationFilters(platform='Nintendo Switch',year_min=2020,year_max=2029,era_basis='edition').matches(game)
    assert not RecommendationFilters(platform='Windows',year_min=2010,year_max=2019,era_basis='edition').matches(game)
    assert not RecommendationFilters(year_min=2010,year_max=2019,era_basis='original').matches(game)


def test_explicit_parent_work_relation_is_required_for_original_year():
    game={'game_id':'wikidata:Q1','metadata':{'wikidata':{'entity_id':'Q1','version_of':['Q2'],
          'original_work_release_date':{'iso':'1999-09-27','parent_entity_id':'Q2','relation':'P629',
                'relation_statement':'source:edition','publication_statement':'source:parent-date',
                'parent_edition_relations':[],'parent_type_facts':[{'value':'http://www.wikidata.org/entity/Q7889'}]}}}}
    assert release_facts(game)['original_year']==1999
    game['metadata']['wikidata']['version_of']=[]
    assert release_facts(game)['original_year'] is None
