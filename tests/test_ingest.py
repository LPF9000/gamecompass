import gzip
import json
import sqlite3
from pathlib import Path

import pytest
from gamecompass.ingest import records, process_ucsd, posted_timestamp, steam_id, user_id, export_recbole, acquire, download


def compressed(path, rows):
    with gzip.open(path, "wt") as f:
        for row in rows:
            f.write(repr(row) + "\n")


def test_source_parser_never_evaluates_code(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("__import__('os').system('false')\n")
    with pytest.raises(ValueError, match="Invalid record"):
        list(records(path))


def test_identifiers_and_dates():
    assert steam_id("0010") == "steam:10"
    assert steam_id("steam:10") == "steam:10"
    with pytest.raises(ValueError):
        steam_id("None")
    assert user_id("alice") != user_id("alice", "other")
    assert "alice" not in user_id("alice")
    assert posted_timestamp("Posted November 5, 2011.") == 1320451200
    assert posted_timestamp("Posted November 5.") is None


def test_process_dedup_missing_metadata_and_export(tmp_path):
    games, reviews = tmp_path / "games.gz", tmp_path / "reviews.gz"
    compressed(games, [{"id": "10", "title": "Game", "genres": ["Action"], "tags": ["Story Rich"]}, {"id": "bad", "title": "Bad"}])
    compressed(reviews, [{"user_id": "a", "reviews": [
        {"item_id": "10", "recommend": True, "posted": "Posted November 5, 2011."},
        {"item_id": "10", "recommend": False, "posted": "Posted November 6, 2011."},
        {"item_id": "20", "recommend": True, "posted": "Posted November 5."},
        {"item_id": "30", "recommend": "yes"}]}])
    database = tmp_path / "population.sqlite"
    stats = process_ucsd(reviews, games, database)
    assert stats["interactions"] == 2
    assert stats["positive_interactions"] == 1
    assert stats["negative_interactions"] == 1
    assert stats["duplicate_interactions"] == 1
    assert stats["conflicting_labels"] == 1
    assert stats["invalid_interactions"] == 1
    with sqlite3.connect(database) as c:
        assert c.execute("select positive from interactions where game_id='steam:10'").fetchone()[0] == 0
        assert c.execute("select title from games where game_id='steam:20'").fetchone()[0] == "Steam app 20"
    assert export_recbole(database, tmp_path / "recbole") == {"inter": 1, "negative.tsv": 1}
    assert "steam:20" in (tmp_path / "recbole/steam.inter").read_text()
    assert "steam:10" not in (tmp_path / "recbole/steam.inter").read_text()


def test_invalid_gzip_and_checksum(tmp_path):
    path = tmp_path / "bad.gz"
    path.write_bytes(b"not gzip")
    with pytest.raises(ValueError, match="SHA-256"):
        download("https://example.invalid/file", path, expected_sha256="0" * 64)
    with pytest.raises(OSError):
        download("https://example.invalid/file", path)


def test_research_license_gate(tmp_path):
    with pytest.raises(ValueError, match="unresolved"):
        acquire(tmp_path)


def test_kaggle_cohort_keeps_complete_histories(tmp_path):
    import zipfile
    from gamecompass.ingest import process_kaggle
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("games.csv", "app_id,title,date_release,win,mac,linux,rating,positive_ratio,user_reviews,price_final,price_original,discount,steam_deck\n10,First,2020-01-01,true,false,false,Positive,90,2,0,0,0,true\n20,Second,2021-01-01,false,true,true,Positive,80,2,0,0,0,true\n")
        z.writestr("games_metadata.json", '{"app_id":10,"description":"Exploration","tags":["Adventure"]}\n')
        z.writestr("users.csv", "user_id,products,reviews\n2,2,2\n3,1,1\n")
        z.writestr("recommendations.csv", "app_id,helpful,funny,date,is_recommended,hours,user_id,review_id\n10,0,0,2022-01-01,true,2.5,2,1\n20,0,0,2022-02-01,false,0.5,2,2\n10,0,0,2022-01-01,true,3.5,3,3\n")
    database = tmp_path / "population.sqlite"
    stats = process_kaggle(archive, database, user_modulus=2)
    assert stats["source_interactions"] == 3
    assert stats["interactions"] == 2
    assert stats["users"] == 1
    assert stats["positive_interactions"] == 1
    assert stats["negative_interactions"] == 1
    assert stats["scope"] == "development user cohort"
    with sqlite3.connect(database) as c:
        assert c.execute("select playtime from interactions where game_id='steam:10'").fetchone()[0] == 2.5
        assert json.loads(c.execute("select platforms from games where game_id='steam:20'").fetchone()[0]) == ["macOS", "Linux"]


def test_download_resumes_only_with_matching_content_range(tmp_path, monkeypatch):
    import io
    import urllib.request
    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Length": "3", "Content-Range": "bytes 3-5/6"}
    destination = tmp_path / "download.bin"
    destination.with_suffix(".bin.part").write_bytes(b"abc")
    seen = []
    def open_mock(request, **kwargs):
        seen.append(request.headers)
        return Response(b"def")
    monkeypatch.setattr(urllib.request, "urlopen", open_mock)
    info = download("https://example.invalid/file", destination)
    assert destination.read_bytes() == b"abcdef"
    assert seen[0]["Range"] == "bytes=3-"
    assert info["bytes"] == 6


def test_download_server_ignoring_range_restarts(tmp_path, monkeypatch):
    import io
    import urllib.request
    class Response(io.BytesIO):
        status = 200
        headers = {"Content-Length": "6"}
    destination = tmp_path / "download.bin"
    destination.with_suffix(".bin.part").write_bytes(b"abc")
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: Response(b"abcdef"))
    download("https://example.invalid/file", destination)
    assert destination.read_bytes() == b"abcdef"


def test_activation_requires_verified_provenance(tmp_path):
    from gamecompass.ingest import activate_kaggle, init_population, KAGGLE_SOURCE
    with pytest.raises(FileNotFoundError):
        activate_kaggle(tmp_path)
    c = init_population(tmp_path / "kaggle-population.sqlite")
    c.close()
    with pytest.raises(ValueError, match="provenance"):
        activate_kaggle(tmp_path)
    with sqlite3.connect(tmp_path / "kaggle-population.sqlite") as c:
        c.execute("insert into provenance values('snapshot',?)", (json.dumps({"source": KAGGLE_SOURCE}),))
    old = init_population(tmp_path / "population.sqlite")
    with old:
        old.execute("insert into provenance values('snapshot',?)", (json.dumps({"source_namespace": "ucsd-au"}),))
    old.close()
    before = (tmp_path / "population.sqlite").read_bytes()
    activate_kaggle(tmp_path)
    assert (tmp_path / "ucsd-research.sqlite").read_bytes() == before
    assert (tmp_path / "population.sqlite").read_bytes() == (tmp_path / "kaggle-population.sqlite").read_bytes()
    assert json.loads((tmp_path / "active-population.json").read_text())["source"]["version"] == 28
