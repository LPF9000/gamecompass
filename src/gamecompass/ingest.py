"""Streaming, provenance-recorded ingestion of public research Steam data.

Raw data and resulting research artifacts are not covered by the code license.
"""
from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

LOG = logging.getLogger(__name__)
SOURCES = {
    "ucsd-reviews": {"url": "https://mcauleylab.ucsd.edu/public_datasets/data/steam/australian_user_reviews.json.gz", "filename": "reviews.json.gz", "license": "No explicit distribution license located; local research only, not redistributable by this project."},
    "ucsd-games": {"url": "https://cseweb.ucsd.edu/~wckang/steam_games.json.gz", "filename": "games.json.gz", "license": "No explicit distribution license located; local research only, not redistributable by this project."},
}
SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
 game_id TEXT PRIMARY KEY, title TEXT NOT NULL, description TEXT DEFAULT '',
 platforms TEXT DEFAULT '[]', genres TEXT DEFAULT '[]', themes TEXT DEFAULT '[]', metadata TEXT DEFAULT '{}');
CREATE TABLE IF NOT EXISTS interactions (
 user_id TEXT NOT NULL, game_id TEXT NOT NULL, positive INTEGER NOT NULL CHECK(positive IN (0,1)),
 timestamp INTEGER, playtime REAL, PRIMARY KEY(user_id,game_id));
CREATE INDEX IF NOT EXISTS interactions_game ON interactions(game_id);
CREATE TABLE IF NOT EXISTS provenance (key TEXT PRIMARY KEY, value TEXT NOT NULL);
PRAGMA user_version=1;
"""


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, *, expected_sha256: str | None = None,
             retries: int = 3, progress: Callable[[int, int | None], None] | None = None) -> dict:
    """Resume a partial HTTP(S) transfer; verify gzip CRC and optional trusted hash.

    A locally computed hash detects subsequent modifications; it is not a source
    authenticity guarantee when the source does not publish a trusted checksum.
    """
    if not url.startswith("https://"):
        raise ValueError("Downloads must use HTTPS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if not destination.exists():
        for attempt in range(retries):
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "GameCompass-research/0.1", "Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                    resumed = offset > 0 and response.status == 206
                    if resumed:
                        content_range = response.headers.get("Content-Range", "")
                        if not content_range.startswith(f"bytes {offset}-"):
                            raise ValueError("Server returned mismatched Content-Range")
                    start = offset if resumed else 0
                    total_header = response.headers.get("Content-Length")
                    total = start + int(total_header) if total_header else None
                    with partial.open("ab" if resumed else "wb") as out:
                        count = start
                        while chunk := response.read(1024 * 1024):
                            out.write(chunk)
                            count += len(chunk)
                            if progress:
                                progress(count, total)
                        out.flush()
                        os.fsync(out.fileno())
                    if total is not None and count != total:
                        raise OSError(f"Incomplete transfer: {count}/{total} bytes")
                partial.replace(destination)
                break
            except (OSError, urllib.error.URLError):
                if attempt + 1 == retries:
                    raise
                time.sleep(min(2 ** attempt, 8))
    digest = sha256(destination)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {destination.name}")
    if destination.name.endswith(".gz"):
        # Reading the complete stream validates gzip's built-in CRC/truncation checks.
        with gzip.open(destination, "rb") as compressed:
            for _ in iter(lambda: compressed.read(1024 * 1024), b""):
                pass
    return {"url": url, "filename": destination.name, "bytes": destination.stat().st_size,
            "sha256": digest, "expected_checksum_verified": expected_sha256 is not None}


def records(path: Path) -> Iterator[dict]:
    """Parse JSON or Python-literal lines. Never execute source code via eval."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    value = ast.literal_eval(line)
                if not isinstance(value, dict):
                    raise ValueError("Record is not an object")
            except (ValueError, SyntaxError, TypeError) as error:
                raise ValueError(f"Invalid record at {path.name}:{line_no}") from error
            yield value


def steam_id(raw: object) -> str:
    text = str(raw).strip()
    if text.startswith("steam:"):
        text = text[6:]
    if not re.fullmatch(r"[0-9]+", text) or int(text) <= 0:
        raise ValueError("Invalid Steam app ID")
    return f"steam:{int(text)}"


def user_id(raw: object, source: str = "ucsd-au") -> str:
    if not isinstance(raw, (str, int)) or not str(raw).strip():
        raise ValueError("Missing source user identifier")
    # Stable pseudonymization, not anonymization or protection against guessing.
    return source + ":" + hashlib.sha256(str(raw).encode()).hexdigest()


def posted_timestamp(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    text = value.removeprefix("Posted ").strip().rstrip(".")
    try:
        return int(datetime.strptime(text, "%B %d, %Y").replace(tzinfo=timezone.utc).timestamp())
    except ValueError:
        # Reviews without years remain unknown. Never invent a year.
        return None


def strings(value: object) -> list[str]:
    return [v for v in value if isinstance(v, str)] if isinstance(value, list) else []


def init_population(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    return connection


def process_ucsd(reviews: Path, games: Path, output: Path) -> dict:
    """Build a complete normalized SQLite snapshot atomically from source streams."""
    temporary = output.with_suffix(".building.sqlite")
    temporary.unlink(missing_ok=True)
    connection = init_population(temporary)
    counts: Counter = Counter()
    observed_review_fields: set[str] = set()
    observed_game_fields: set[str] = set()
    try:
        with connection:
            for raw in records(games):
                counts["metadata_rows"] += 1
                observed_game_fields.update(raw)
                try:
                    gid = steam_id(raw.get("id"))
                except ValueError:
                    counts["invalid_metadata_ids"] += 1
                    continue
                title = raw.get("title") or raw.get("app_name")
                if not isinstance(title, str) or not title.strip():
                    counts["missing_titles"] += 1
                    continue
                genres = strings(raw.get("genres"))
                tags = strings(raw.get("tags"))
                # This release does not reliably provide supported operating systems.
                # Platform stays unknown; do not infer Windows from Steam membership.
                metadata = {key: raw[key] for key in ("release_date", "developer", "publisher", "specs", "early_access", "url") if key in raw}
                metadata.update({"external_ids": {"steam": gid[6:]}, "tags": tags,
                                 "source": "ucsd-steam-games", "identity_confidence": 1.0})
                description = raw.get("description", "")
                if not isinstance(description, str):
                    description = ""
                connection.execute("INSERT OR REPLACE INTO games VALUES(?,?,?,?,?,?,?)",
                                   (gid, title.strip(), description, "[]", json.dumps(genres), "[]", json.dumps(metadata)))
            for raw in records(reviews):
                counts["source_users"] += 1
                try:
                    uid = user_id(raw.get("user_id"))
                except ValueError:
                    counts["invalid_users"] += 1
                    continue
                reviews_list = raw.get("reviews")
                if not isinstance(reviews_list, list):
                    raise ValueError("UCSD review row lacks reviews array")
                for review in reviews_list:
                    counts["source_interactions"] += 1
                    if not isinstance(review, dict):
                        counts["invalid_interactions"] += 1
                        continue
                    observed_review_fields.update(review)
                    try:
                        gid = steam_id(review.get("item_id"))
                        if type(review.get("recommend")) is not bool:
                            raise ValueError("Recommendation label must be boolean")
                    except ValueError:
                        counts["invalid_interactions"] += 1
                        continue
                    stamp = posted_timestamp(review.get("posted"))
                    if stamp is None:
                        counts["unknown_timestamps"] += 1
                    old = connection.execute("SELECT positive,timestamp FROM interactions WHERE user_id=? AND game_id=?", (uid, gid)).fetchone()
                    if old:
                        counts["duplicate_interactions"] += 1
                        if old[0] != int(review["recommend"]):
                            counts["conflicting_labels"] += 1
                        # Retain most recent known timestamp. Identical/unknown ties
                        # retain first source occurrence for deterministic behavior.
                        if stamp is None or (old[1] is not None and stamp <= old[1]):
                            continue
                    connection.execute("INSERT OR REPLACE INTO interactions VALUES(?,?,?,?,?)",
                                       (uid, gid, int(review["recommend"]), stamp, None))
                    connection.execute("INSERT OR IGNORE INTO games(game_id,title,metadata) VALUES(?,?,?)",
                                       (gid, f"Steam app {gid[6:]}", json.dumps({"source": "ucsd-au-review", "missing_title": True, "external_ids": {"steam": gid[6:]}})))
            counts.update({"interactions": connection.execute("SELECT COUNT(*) FROM interactions").fetchone()[0],
                           "users": connection.execute("SELECT COUNT(DISTINCT user_id) FROM interactions").fetchone()[0],
                           "interaction_games": connection.execute("SELECT COUNT(DISTINCT game_id) FROM interactions").fetchone()[0],
                           "catalog_games": connection.execute("SELECT COUNT(*) FROM games").fetchone()[0],
                           "positive_interactions": connection.execute("SELECT COUNT(*) FROM interactions WHERE positive=1").fetchone()[0],
                           "negative_interactions": connection.execute("SELECT COUNT(*) FROM interactions WHERE positive=0").fetchone()[0],
                           "known_timestamps": connection.execute("SELECT COUNT(*) FROM interactions WHERE timestamp IS NOT NULL").fetchone()[0]})
            stats = dict(counts)
            stats["sparsity"] = 1 - stats["interactions"] / max(1, stats["users"] * stats["interaction_games"])
            stats["review_fields"] = sorted(observed_review_fields)
            stats["game_fields"] = sorted(observed_game_fields)
            stats["source_namespace"] = "ucsd-au"
            stats["schema_version"] = 1
            stats["license_status"] = "local-research-only; redistribution unresolved"
            stats["inputs"] = [{"filename": p.name, "sha256": sha256(p), "bytes": p.stat().st_size} for p in (reviews, games)]
            connection.execute("INSERT OR REPLACE INTO provenance VALUES(?,?)", ("snapshot", json.dumps(stats)))
        connection.close()
        temporary.replace(output)
    except BaseException:
        connection.close()
        raise
    atomic_json(output.parent / "dataset-stats.json", stats)
    return stats


def export_recbole(database: Path, directory: Path, dataset: str = "steam") -> dict:
    """Export positives separately from explicit negatives, preserving timestamps.

    Missing timestamps are exported as 0, with missingness in a companion file;
    temporal evaluators MUST exclude timestamp==0. No raw review text is exported.
    """
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", dataset):
        raise ValueError("Invalid dataset name")
    directory.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    totals = {}
    try:
        for label, suffix in [(1, "inter"), (0, "negative.tsv")]:
            path = directory / f"{dataset}.{suffix}"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="\t", lineterminator="\n")
                writer.writerow(["user_id:token", "item_id:token", "timestamp:float"])
                count = 0
                for uid, gid, stamp in connection.execute("SELECT user_id,game_id,timestamp FROM interactions WHERE positive=? ORDER BY user_id,game_id", (label,)):
                    writer.writerow([uid, gid, stamp or 0])
                    count += 1
            totals[suffix] = count
    finally:
        connection.close()
    atomic_json(directory / "export.json", {"schema": 1, "dataset": dataset, "counts": totals,
                "missing_timestamp": 0, "negative_semantics": "Explicit non-recommendations; excluded from positive atomic file."})
    return totals


def acquire(data_dir: Path, *, research_only: bool = False) -> dict:
    """Acquire UCSD research snapshot. Caller must explicitly select research use."""
    if not research_only:
        raise ValueError("UCSD application/redistribution permission is unresolved. Pass research_only=True for local research; do not redistribute data or weights.")
    prior_path = data_dir / "download-manifest.json"
    manifest = json.loads(prior_path.read_text()) if prior_path.exists() else {}
    for name, source in SOURCES.items():
        LOG.info("Acquiring %s", name)
        destination = data_dir / "raw" / source["filename"]
        recorded = manifest.get(name, {})
        expected = recorded.get("sha256") if recorded.get("url") == source["url"] else None
        manifest[name] = {**download(source["url"], destination, expected_sha256=expected), "license": source["license"]}
        atomic_json(data_dir / "download-manifest.json", manifest)
    stats = process_ucsd(data_dir / "raw/reviews.json.gz", data_dir / "raw/games.json.gz", data_dir / "population.sqlite")
    export_recbole(data_dir / "population.sqlite", data_dir / "recbole/steam")
    return stats

KAGGLE_SOURCE = {
    "ref": "antonkozyriev/game-recommendations-on-steam", "version": 28,
    "url": "https://www.kaggle.com/api/v1/datasets/download/antonkozyriev/game-recommendations-on-steam?datasetVersionNumber=28",
    "license": "CC0: Public Domain (publisher declaration in Kaggle API)",
    "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
    "listing_url": "https://www.kaggle.com/api/v1/datasets/list?search=game-recommendations-on-steam",
}


def process_kaggle(archive: Path, output: Path, *, user_modulus: int = 32,
                   progress: Callable[[int, int], None] | None = None) -> dict:
    """Stream every source review and retain all rows for a deterministic user cohort.

    IDs are numeric in this release. Cohort is source_user_id % user_modulus==0,
    a systematic sample, NOT a claim of a statistically random sample. modulus=1
    processes the full dataset. A bounded development cohort is explicitly labeled.
    """
    import io
    import zipfile
    if not isinstance(user_modulus, int) or user_modulus < 1:
        raise ValueError("user_modulus must be a positive integer")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".building.sqlite")
    temporary.unlink(missing_ok=True)
    connection = init_population(temporary)
    connection.execute("PRAGMA synchronous=NORMAL")  # Interrupted builds are discarded; final replacement is atomic.
    counts: Counter = Counter()
    started = time.monotonic()
    batch = []
    cached_users: dict[str, str] = {}
    day_cache: dict[str, int] = {}
    try:
        with zipfile.ZipFile(archive) as z, connection:
            required = {"games.csv", "games_metadata.json", "recommendations.csv", "users.csv"}
            if not required.issubset(z.namelist()):
                raise ValueError("Kaggle v28 archive lacks required members")
            members = [{"filename": item.filename, "uncompressed_bytes": item.file_size, "crc32": f"{item.CRC:08x}"} for item in z.infolist()]
            with z.open("games.csv") as f:
                for raw in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                    gid = steam_id(raw["app_id"])
                    platforms = [name for name, field in [("Windows", "win"), ("macOS", "mac"), ("Linux", "linux")] if raw[field].lower() == "true"]
                    metadata = {"source": "kaggle-steam-v28", "external_ids": {"steam": gid[6:]}, "release_date": raw.get("date_release"), "identity_confidence": 1.0}
                    connection.execute("INSERT OR REPLACE INTO games VALUES(?,?,?,?,?,?,?)", (gid, raw["title"], "", json.dumps(platforms), "[]", "[]", json.dumps(metadata)))
                    counts["source_catalog_rows"] += 1
            with z.open("games_metadata.json") as f:
                for line in io.TextIOWrapper(f, encoding="utf-8"):
                    raw = json.loads(line)
                    gid = steam_id(raw["app_id"])
                    row = connection.execute("SELECT metadata FROM games WHERE game_id=?", (gid,)).fetchone()
                    if row:
                        metadata = json.loads(row[0])
                        metadata["tags"] = strings(raw.get("tags"))
                        connection.execute("UPDATE games SET description=?,metadata=? WHERE game_id=?", (str(raw.get("description") or ""), json.dumps(metadata), gid))
                    counts["source_metadata_rows"] += 1
            with z.open("users.csv") as f:
                user_reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
                user_fields = next(user_reader)
                if user_fields != ["user_id", "products", "reviews"]:
                    raise ValueError("Unexpected Kaggle user schema")
                counts["source_users"] = sum(1 for _ in user_reader)
            with z.open("recommendations.csv") as f:
                reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
                fields = next(reader)
                if fields != ["app_id", "helpful", "funny", "date", "is_recommended", "hours", "user_id", "review_id"]:
                    raise ValueError(f"Unexpected Kaggle review schema: {fields}")
                for row in reader:
                    counts["source_interactions"] += 1
                    if len(row) != len(fields):
                        raise ValueError("Malformed Kaggle recommendation row")
                    if row[4] not in ("true", "false"):
                        raise ValueError("Non-boolean source recommendation label")
                    counts["source_positive" if row[4] == "true" else "source_negative"] += 1
                    if int(row[6]) % user_modulus:
                        continue
                    if row[6] not in cached_users:
                        cached_users[row[6]] = user_id(row[6], "kaggle-steam-v28")
                    if row[3] not in day_cache:
                        day_cache[row[3]] = int(datetime.strptime(row[3], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
                    hours = float(row[5])
                    if not 0 <= hours < float("inf"):
                        raise ValueError("Invalid playtime")
                    batch.append((cached_users[row[6]], steam_id(row[0]), int(row[4] == "true"), day_cache[row[3]], hours))
                    counts["selected_source_interactions"] += 1
                    if len(batch) >= 10000:
                        connection.executemany("INSERT INTO interactions VALUES(?,?,?,?,?) ON CONFLICT(user_id,game_id) DO UPDATE SET positive=excluded.positive,timestamp=excluded.timestamp,playtime=excluded.playtime WHERE excluded.timestamp>interactions.timestamp", batch)
                        batch.clear()
                        if progress:
                            progress(counts["source_interactions"], counts["selected_source_interactions"])
                if batch:
                    connection.executemany("INSERT INTO interactions VALUES(?,?,?,?,?) ON CONFLICT(user_id,game_id) DO UPDATE SET positive=excluded.positive,timestamp=excluded.timestamp,playtime=excluded.playtime WHERE excluded.timestamp>interactions.timestamp", batch)
            counts.update({"users": connection.execute("SELECT COUNT(DISTINCT user_id) FROM interactions").fetchone()[0],
                           "interactions": connection.execute("SELECT COUNT(*) FROM interactions").fetchone()[0],
                           "interaction_games": connection.execute("SELECT COUNT(DISTINCT game_id) FROM interactions").fetchone()[0],
                           "catalog_games": connection.execute("SELECT COUNT(*) FROM games").fetchone()[0],
                           "positive_interactions": connection.execute("SELECT COUNT(*) FROM interactions WHERE positive=1").fetchone()[0],
                           "negative_interactions": connection.execute("SELECT COUNT(*) FROM interactions WHERE positive=0").fetchone()[0]})
            counts["duplicate_interactions"] = counts["selected_source_interactions"] - counts["interactions"]
            stats = dict(counts)
            import sys
            try:
                import resource
                peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                rss_mib = peak_rss / (1024 * 1024 if sys.platform == "darwin" else 1024)
            except ImportError:  # Windows reports peak working set via psutil.
                import psutil
                memory = psutil.Process().memory_info()
                rss_mib = getattr(memory, "peak_wset", memory.rss) / (1024 * 1024)
            span = connection.execute("SELECT MIN(timestamp), MAX(timestamp), MIN(playtime), MAX(playtime) FROM interactions").fetchone()
            stats.update({"peak_process_rss_mib": round(rss_mib, 2), "timestamp_min": span[0], "timestamp_max": span[1], "playtime_hours_min": span[2], "playtime_hours_max": span[3], "schema_version": 1, "source": KAGGLE_SOURCE, "scope": "full" if user_modulus == 1 else "development user cohort",
                          "selection": f"numeric source user_id % {user_modulus} == 0; complete selected histories", "user_modulus": user_modulus,
                          "sparsity": 1 - counts["interactions"] / max(1, counts["users"] * counts["interaction_games"]),
                          "observed_interaction_fields": fields, "members": members,
                          "archive": {"bytes": archive.stat().st_size, "sha256": sha256(archive)},
                          "processing_seconds": round(time.monotonic() - started, 3)})
            connection.execute("INSERT OR REPLACE INTO provenance VALUES(?,?)", ("snapshot", json.dumps(stats)))
        connection.close()
        temporary.replace(output)
    except BaseException:
        connection.close()
        raise
    atomic_json(output.with_suffix(".stats.json"), stats)
    return stats


def acquire_kaggle(data_dir: Path, *, user_modulus: int = 32,
                   progress: Callable[[int, int], None] | None = None) -> dict:
    """Acquire the publisher's CC0-declared Steam dataset release 28."""
    archive = data_dir / "raw/kaggle-steam-v28.zip"
    prior_path = data_dir / "kaggle-download-manifest.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}
    expected = prior.get("sha256") if prior.get("url") == KAGGLE_SOURCE["url"] else None
    manifest = {**download(KAGGLE_SOURCE["url"], archive, expected_sha256=expected), **KAGGLE_SOURCE}
    atomic_json(data_dir / "kaggle-download-manifest.json", manifest)
    stats = process_kaggle(archive, data_dir / "kaggle-population.sqlite", user_modulus=user_modulus, progress=progress)
    export_recbole(data_dir / "kaggle-population.sqlite", data_dir / "recbole/kaggle", dataset="kaggle")
    return stats


def activate_kaggle(data_dir: Path) -> Path:
    """Promote a successfully built CC0 snapshot while preserving research data.

    Explicit action, never called implicitly during acquisition. Source and target
    must not be undergoing writes. The metadata provenance remains in SQLite.
    """
    import shutil
    source = data_dir / "kaggle-population.sqlite"
    if not source.is_file():
        raise FileNotFoundError("Acquire and process the Kaggle snapshot first")
    with sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True) as c:
        if c.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Cannot activate an invalid population snapshot")
        snapshot_row = c.execute("SELECT value FROM provenance WHERE key='snapshot'").fetchone()
        if snapshot_row is None or json.loads(snapshot_row[0]).get("source", {}).get("ref") != KAGGLE_SOURCE["ref"]:
            raise ValueError("Cannot activate a snapshot with unverified source provenance")
    destination = data_dir / "population.sqlite"
    if destination.exists():
        with sqlite3.connect(f"file:{destination.resolve()}?mode=ro", uri=True) as c:
            row = c.execute("SELECT value FROM provenance WHERE key='snapshot'").fetchone()
            is_ucsd = row is not None and json.loads(row[0]).get("source_namespace") == "ucsd-au"
        if is_ucsd:
            backup = data_dir / "ucsd-research.sqlite"
            if backup.exists() and sha256(backup) != sha256(destination):
                raise ValueError("Existing UCSD backup differs; preserve or move it before activating")
            backup_tmp = backup.with_suffix(".tmp")
            shutil.copyfile(destination, backup_tmp)
            backup_tmp.replace(backup)
    stage = data_dir / "population.activating.sqlite"
    shutil.copyfile(source, stage)
    stage.replace(destination)
    atomic_json(data_dir / "active-population.json", {"source": KAGGLE_SOURCE, "sha256": sha256(destination), "database": destination.name})
    return destination
