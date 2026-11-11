import pytest
import requests

from gamecompass.catalog import CatalogError, IGDBClient, enrich_steam_game, resolve_candidates
from gamecompass.store import Store


def candidate(igdb_id=1, app_id="10", name="Game"):
    return {"id": igdb_id, "name": name, "platforms": [{"name": "PlayStation 5"}],
            "external_games": [{"uid": app_id, "external_game_source": {"name": "Steam"}}]}


def test_names_never_merge():
    result = resolve_candidates({"game_id": "steam:10", "title": "Game"},
                                [{"id": 1, "name": "Game"}, candidate(2, "11", "Game")])
    assert result.status == "review" and result.igdb_id is None
    assert len(result.candidates) == 2


def test_exact_id_and_ambiguity():
    source = {"game_id": "steam:10", "title": "Game"}
    assert resolve_candidates(source, [candidate()]).igdb_id == 1
    assert resolve_candidates(source, [candidate(), candidate()]).igdb_id == 1
    assert resolve_candidates(source, [candidate(), candidate(2)]).status == "review"
    assert resolve_candidates(source, []).status == "unmatched"
    assert resolve_candidates(source, [{"id": 1, "external_games": [{"uid": "10", "external_game_source": 1}]}]).status == "review"
    legacy = {"id": 1, "external_games": [{"uid": "10", "category": 1}]}
    assert resolve_candidates(source, [legacy]).status == "matched"


def test_variant_conflict_and_bad_source():
    source = {"game_id": "steam:10", "metadata": {"game_type": "Main Game"}}
    row = candidate()
    row["game_type"] = {"type": "Remake"}
    assert resolve_candidates(source, [row]).status == "review"
    with pytest.raises(ValueError):
        resolve_candidates({"game_id": 'steam:10; where id>0;'}, [])


class Response:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class Session:
    def __init__(self):
        self.calls = []
        self.payloads = [[{"id": 42, "name": "Steam"}],
                         [{"uid": "10", "game": 1, "external_game_source": 42}], [candidate()]]

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return Response(self.payloads.pop(0))


def test_discovered_source_id_and_header_credentials(monkeypatch):
    monkeypatch.setattr("gamecompass.catalog.time.sleep", lambda _: None)
    session = Session()
    client = IGDBClient(client_id="client", access_token="secret", session=session)
    assert client.steam_candidates("10")[0]["id"] == 1
    assert "external_game_source = 42" in session.calls[1][1]["data"]
    assert all("secret" not in url for url, _ in session.calls)
    assert session.calls[0][1]["headers"]["Authorization"] == "Bearer secret"


def test_cache_gate_and_persistence(tmp_path):
    store = Store(tmp_path)
    store.upsert_game({"game_id": "steam:10", "title": "Original title", "platforms": ["Windows"]})

    class Client:
        def steam_candidates(self, app_id):
            assert app_id == "10"
            return [candidate()]

    with pytest.raises(CatalogError, match="acknowledgement"):
        enrich_steam_game(store, "steam:10", Client())
    decision = enrich_steam_game(store, "steam:10", Client(), acknowledge_provider_terms=True)
    assert decision.status == "matched"
    game = Store(tmp_path).game("steam:10")
    assert game["title"] == "Original title"
    assert game["platforms"] == ["PlayStation 5", "Windows"]
    assert game["metadata"]["igdb_terms"]["redistribution_authorized"] is False


def test_missing_credentials_and_sanitized_failure(monkeypatch):
    monkeypatch.delenv("IGDB_CLIENT_ID", raising=False)
    monkeypatch.delenv("IGDB_ACCESS_TOKEN", raising=False)
    with pytest.raises(CatalogError, match="IGDB_CLIENT_ID"):
        IGDBClient()
    monkeypatch.setattr("gamecompass.catalog.time.sleep", lambda _: None)

    class Broken:
        def post(self, *args, **kwargs):
            raise requests.ConnectionError("secret token should never appear")

    with pytest.raises(CatalogError) as error:
        IGDBClient(client_id="c", access_token="secret", session=Broken()).steam_candidates("10")
    assert "secret" not in str(error.value)
