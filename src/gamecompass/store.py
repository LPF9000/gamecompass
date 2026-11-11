"""Transactional local profiles, explicit preferences, and an offline game catalog.

Population interactions never enter this database. Import/export does not contain
credentials, and local preferences are never included in training implicitly.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

STATUSES = ("unplayed", "owned", "played", "completed", "abandoned", "wishlist")
REASONS = ("Story", "Characters", "Exploration", "World design", "Combat", "Progression", "Atmosphere", "Difficulty", "Pacing", "Repetition", "Creativity")
RATING_LABELS = {2: "Loved it", 1: "Liked it", 0: "Neutral", -1: "Disliked it", -2: "Hated it"}
GAME_SCHEMA = """CREATE TABLE IF NOT EXISTS games (
 game_id TEXT PRIMARY KEY, title TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
 platforms TEXT NOT NULL DEFAULT '[]', genres TEXT NOT NULL DEFAULT '[]',
 themes TEXT NOT NULL DEFAULT '[]', metadata TEXT NOT NULL DEFAULT '{}')"""


def _game_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    for field, fallback in (("platforms", []), ("genres", []), ("themes", []), ("metadata", {})):
        try:
            result[field] = json.loads(result[field])
        except (ValueError, TypeError):
            result[field] = fallback
    return result


class Store:
    """Each method owns its connection, so reruns and background jobs are safe."""

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / "local.sqlite"
        with self.connection() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version > 1:
                raise ValueError("Local database was created by a newer application")
            db.execute(GAME_SCHEMA)
            db.executescript("""
            CREATE TABLE IF NOT EXISTS local_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS profiles(id TEXT PRIMARY KEY, name TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS feedback (
                profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
                game_id TEXT NOT NULL REFERENCES games(game_id), rating INTEGER CHECK(rating BETWEEN -2 AND 2),
                status TEXT NOT NULL CHECK(status IN ('unplayed','owned','played','completed','abandoned','wishlist')),
                not_interested INTEGER NOT NULL DEFAULT 0 CHECK(not_interested IN (0,1)),
                reasons TEXT NOT NULL DEFAULT '[]', updated REAL NOT NULL,
                PRIMARY KEY(profile_id,game_id));
            CREATE TABLE IF NOT EXISTS implicit_signals (
                profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
                game_id TEXT NOT NULL REFERENCES games(game_id), provider TEXT NOT NULL,
                playtime_minutes REAL NOT NULL DEFAULT 0, metadata TEXT NOT NULL DEFAULT '{}',
                updated REAL NOT NULL, PRIMARY KEY(profile_id,game_id,provider));
            CREATE TABLE IF NOT EXISTS connections (
                profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
                provider TEXT NOT NULL, imported_at REAL NOT NULL, game_count INTEGER NOT NULL,
                PRIMARY KEY(profile_id,provider));
            CREATE TABLE IF NOT EXISTS personal_evaluations (
                profile_id TEXT REFERENCES profiles(id) ON DELETE CASCADE, report TEXT, created REAL);
            CREATE TABLE IF NOT EXISTS outcomes (
                profile_id TEXT REFERENCES profiles(id) ON DELETE CASCADE,
                game_id TEXT, kind TEXT, value INTEGER, created REAL);
            CREATE INDEX IF NOT EXISTS games_title ON games(title COLLATE NOCASE);
            PRAGMA user_version=1;
            """)
        self.sync_catalog()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30)
        self.path.chmod(0o600)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def sync_catalog(self) -> int:
        """Copy population metadata, never population users, into the local cache."""
        source = self.data_dir / "population.sqlite"
        if not source.exists():
            return 0
        stat = source.stat()
        signature = f"{stat.st_mtime_ns}:{stat.st_size}"
        with self.connection() as db:
            previous = db.execute("SELECT value FROM local_metadata WHERE key='catalog_signature'").fetchone()
            if previous and previous[0] == signature:
                return 0
        with sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True) as population:
            if not population.execute("SELECT 1 FROM sqlite_master WHERE name='games'").fetchone():
                return 0
            with self.connection() as db:
                cursor = population.execute("SELECT game_id,title,description,platforms,genres,themes,metadata FROM games")
                count = 0
                while batch := cursor.fetchmany(2000):
                    db.executemany("""INSERT INTO games VALUES (?,?,?,?,?,?,?) ON CONFLICT(game_id) DO UPDATE SET
                    description=CASE WHEN games.description='' THEN excluded.description ELSE games.description END,
                    platforms=CASE WHEN games.platforms='[]' THEN excluded.platforms ELSE games.platforms END,
                    genres=CASE WHEN games.genres='[]' THEN excluded.genres ELSE games.genres END,
                    themes=CASE WHEN games.themes='[]' THEN excluded.themes ELSE games.themes END,
                    metadata=CASE WHEN games.metadata='{}' THEN excluded.metadata ELSE games.metadata END""", batch)
                    count += len(batch)
                db.execute("INSERT OR REPLACE INTO local_metadata VALUES ('catalog_signature',?)", (signature,))
                return count

    def catalog_additions(self, known_ids: set[str]) -> list[dict[str, Any]]:
        """Return new cached games for metadata-only inference without population retraining."""
        with self.connection() as db:
            ids = [row[0] for row in db.execute('SELECT game_id FROM games') if row[0] not in known_ids]
            return [_game_dict(db.execute('SELECT * FROM games WHERE game_id=?',(g,)).fetchone()) for g in ids]

    def catalog_games(self) -> list[dict[str, Any]]:
        """Decoded local catalog for refinements and metadata-only candidate retrieval."""
        with self.connection() as db:
            return [_game_dict(row) for row in db.execute('SELECT * FROM games ORDER BY game_id')]

    def profiles(self) -> list[dict[str, Any]]:
        with self.connection() as db:
            return [dict(row) for row in db.execute("SELECT * FROM profiles ORDER BY created,id")]

    @staticmethod
    def _name(name: str) -> str:
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80:
            raise ValueError("Profile name must contain 1 to 80 characters")
        return name.strip()

    def create_profile(self, name: str) -> str:
        name = self._name(name)
        profile_id = uuid.uuid4().hex
        with self.connection() as db:
            db.execute("INSERT INTO profiles VALUES (?,?,?)", (profile_id, name, time.time()))
        return profile_id

    @staticmethod
    def _profile(db: sqlite3.Connection, profile_id: str) -> sqlite3.Row:
        row = db.execute("SELECT * FROM profiles WHERE id=?", (profile_id,)).fetchone()
        if row is None:
            raise ValueError("Profile does not exist")
        return row

    def delete_profile(self, profile_id: str) -> None:
        with self.connection() as db:
            self._profile(db, profile_id)
            db.execute("DELETE FROM profiles WHERE id=?", (profile_id,))
            # VACUUM is deliberately separate from transaction, removes deleted
            # profile pages from the local file instead of leaving recoverable rows.
        with sqlite3.connect(self.path) as db:
            db.execute("VACUUM")

    @staticmethod
    def _validated_game(game: dict[str, Any]) -> tuple[Any, ...]:
        if not isinstance(game, dict):
            raise ValueError("Game must be an object")
        game_id, title = game.get("game_id"), game.get("title")
        if not isinstance(game_id, str) or not game_id.strip() or len(game_id) > 160:
            raise ValueError("Game needs a valid canonical game_id")
        if not isinstance(title, str) or not title.strip() or len(title) > 1000:
            raise ValueError("Game needs a title")
        description = game.get("description", "")
        if not isinstance(description, str) or len(description) > 100000:
            raise ValueError("Invalid game description")
        values: list[Any] = [game_id, title.strip(), description]
        for field in ("platforms", "genres", "themes"):
            value = game.get(field, [])
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                raise ValueError(f"{field} must be a list of strings")
            values.append(json.dumps(value))
        metadata = game.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        values.append(json.dumps(metadata))
        return tuple(values)

    def upsert_game(self, game: dict[str, Any]) -> None:
        values = self._validated_game(game)
        with self.connection() as db:
            db.execute("""INSERT INTO games VALUES (?,?,?,?,?,?,?) ON CONFLICT(game_id) DO UPDATE SET
            title=excluded.title, description=CASE WHEN excluded.description='' THEN games.description ELSE excluded.description END,
            platforms=CASE WHEN excluded.platforms='[]' THEN games.platforms ELSE excluded.platforms END,
            genres=CASE WHEN excluded.genres='[]' THEN games.genres ELSE excluded.genres END,
            themes=CASE WHEN excluded.themes='[]' THEN games.themes ELSE excluded.themes END,
            metadata=CASE WHEN excluded.metadata='{}' THEN games.metadata ELSE excluded.metadata END""", values)

    def search(self, query: str, limit: int = 30) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 1000))
        # Escape wildcard characters so title input remains literal.
        query = query.strip()[:300]
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self.connection() as db:
            rows = db.execute("""SELECT * FROM games WHERE title LIKE ? ESCAPE '\\' OR metadata LIKE ? ESCAPE '\\'
            ORDER BY CASE WHEN title LIKE ? ESCAPE '\\' THEN 0 ELSE 1 END, title COLLATE NOCASE,game_id LIMIT ?""",
                              (f"%{escaped}%", f"%{escaped}%", f"{escaped}%", limit))
            return [_game_dict(row) for row in rows]

    def game(self, game_id: str) -> dict[str, Any] | None:
        with self.connection() as db:
            row = db.execute("SELECT * FROM games WHERE game_id=?", (game_id,)).fetchone()
            return _game_dict(row) if row else None

    def platforms(self) -> list[str]:
        with self.connection() as db:
            rows = db.execute("SELECT DISTINCT platforms FROM games")
            return sorted({platform for row in rows for platform in json.loads(row[0])})

    def catalog_count(self) -> int:
        with self.connection() as db:
            return db.execute("SELECT COUNT(*) FROM games").fetchone()[0]

    def feedback(self, profile_id: str) -> list[dict[str, Any]]:
        with self.connection() as db:
            self._profile(db, profile_id)
            rows = db.execute("""SELECT f.*,g.title FROM feedback f JOIN games g USING(game_id)
            WHERE profile_id=? ORDER BY updated DESC,game_id""", (profile_id,))
            result = []
            for row in rows:
                item = dict(row)
                item["reasons"] = json.loads(item["reasons"])
                item["not_interested"] = bool(item["not_interested"])
                result.append(item)
            return result

    @staticmethod
    def _validated_feedback(rating: int | None, status: str, not_interested: bool,
                            reasons: list[str] | None) -> tuple[Any, ...]:
        if rating is not None and (type(rating) is not int or rating not in RATING_LABELS):
            raise ValueError("Rating must be an integer from -2 to 2 or null")
        if status not in STATUSES:
            raise ValueError("Unsupported activity state")
        if type(not_interested) is not bool:
            raise ValueError("not_interested must be true or false")
        reasons = [] if reasons is None else reasons
        if not isinstance(reasons, list) or any(reason not in REASONS for reason in reasons):
            raise ValueError("Unsupported preference reason")
        return rating, status, int(not_interested), json.dumps(sorted(set(reasons)))

    def set_feedback(self, profile_id: str, game_id: str, rating: int | None,
                     status: str = "unplayed", not_interested: bool = False,
                     reasons: list[str] | None = None) -> None:
        values = self._validated_feedback(rating, status, not_interested, reasons)
        with self.connection() as db:
            self._profile(db, profile_id)
            if not db.execute("SELECT 1 FROM games WHERE game_id=?", (game_id,)).fetchone():
                raise ValueError("Unknown game; add it to the catalog first")
            db.execute("""INSERT INTO feedback VALUES (?,?,?,?,?,?,?) ON CONFLICT(profile_id,game_id)
            DO UPDATE SET rating=excluded.rating,status=excluded.status,not_interested=excluded.not_interested,
            reasons=excluded.reasons,updated=excluded.updated""", (profile_id, game_id, *values, time.time()))

    def implicit(self, profile_id: str) -> list[dict[str, Any]]:
        with self.connection() as db:
            self._profile(db, profile_id)
            return [dict(row) for row in db.execute("SELECT * FROM implicit_signals WHERE profile_id=?", (profile_id,))]

    def profile_connections(self, profile_id: str) -> list[dict[str, Any]]:
        with self.connection() as db:
            self._profile(db, profile_id)
            return [dict(row) for row in db.execute("SELECT * FROM connections WHERE profile_id=?", (profile_id,))]

    def import_library(self, profile_id: str, games: list[dict[str, Any]], provider: str = "steam") -> int:
        """Import weaker activity signals without changing any explicit preference."""
        if not isinstance(games, list) or not 1 <= len(provider) <= 60:
            raise ValueError("Invalid library import")
        validated = []
        for game in games:
            values = self._validated_game(game)
            playtime = float(game.get("playtime_minutes", 0))
            if not 0 <= playtime < 100000000:
                raise ValueError("Invalid playtime")
            validated.append((values, playtime, game.get("achievements", {})))
        with self.connection() as db:
            self._profile(db, profile_id)
            for values, playtime, achievements in validated:
                db.execute("INSERT OR IGNORE INTO games VALUES (?,?,?,?,?,?,?)", values)
                db.execute("""INSERT INTO implicit_signals VALUES (?,?,?,?,?,?) ON CONFLICT(profile_id,game_id,provider)
                DO UPDATE SET playtime_minutes=excluded.playtime_minutes,metadata=excluded.metadata,updated=excluded.updated""",
                           (profile_id, values[0], provider, playtime, json.dumps({"achievements": achievements}), time.time()))
                db.execute("INSERT OR IGNORE INTO feedback VALUES (?,?,NULL,?,0,'[]',?)",
                           (profile_id, values[0], "played" if playtime > 0 else "owned", time.time()))
            db.execute("""INSERT INTO connections VALUES (?,?,?,?) ON CONFLICT(profile_id,provider)
            DO UPDATE SET imported_at=excluded.imported_at,game_count=excluded.game_count""",
                       (profile_id, provider, time.time(), len(validated)))
        return len(validated)

    def personal_history(self, profile_id: str) -> dict[str, list[dict[str, Any]]]:
        with self.connection() as db:
            self._profile(db, profile_id)
            evaluations = [{"report": json.loads(row["report"]), "created": row["created"]} for row in
                           db.execute("SELECT report,created FROM personal_evaluations WHERE profile_id=? ORDER BY created", (profile_id,))]
            outcomes = [dict(row) for row in db.execute("SELECT game_id,kind,value,created FROM outcomes WHERE profile_id=? ORDER BY created", (profile_id,))]
        return {"personal_evaluations": evaluations, "outcomes": outcomes}

    def export_profile(self, profile_id: str) -> dict[str, Any]:
        with self.connection() as db:
            profile = self._profile(db, profile_id)
            name = profile["name"]
        feedback = self.feedback(profile_id)
        implicit = self.implicit(profile_id)
        history = self.personal_history(profile_id)
        game_ids = {row["game_id"] for row in feedback + implicit + history["outcomes"]}
        return {"schema_version": 1, "profile": {"name": name},
                "games": [self.game(game_id) for game_id in sorted(game_ids)],
                "feedback": [{key: row[key] for key in ("game_id", "rating", "status", "not_interested", "reasons")} for row in feedback],
                "implicit_signals": [{key: row[key] for key in ("game_id", "provider", "playtime_minutes", "metadata")} for row in implicit],
                **history}

    def import_profile(self, payload: dict[str, Any]) -> str:
        """Validate entire payload before an atomic import into a fresh profile."""
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("Unsupported profile export version")
        profile = payload.get("profile", {})
        if not isinstance(profile, dict):
            raise ValueError("Profile must be an object")
        name = self._name(profile.get("name"))
        games = payload.get("games", [])
        feedback = payload.get("feedback", [])
        implicit = payload.get("implicit_signals", [])
        if any(not isinstance(items, list) for items in (games, feedback, implicit)):
            raise ValueError("Invalid profile collections")
        validated_games = [self._validated_game(game) for game in games]
        game_ids = {game[0] for game in validated_games}
        validated_feedback = []
        for row in feedback:
            if not isinstance(row, dict):
                raise ValueError("Feedback must be an object")
            if row.get("game_id") not in game_ids:
                raise ValueError("Feedback references game absent from export")
            values = self._validated_feedback(row.get("rating"), row.get("status", "unplayed"),
                                             row.get("not_interested", False), row.get("reasons", []))
            validated_feedback.append((row["game_id"], *values))
        validated_implicit = []
        for row in implicit:
            if not isinstance(row, dict):
                raise ValueError("Library signal must be an object")
            if row.get("game_id") not in game_ids:
                raise ValueError("Library references game absent from export")
            provider = row.get("provider", "manual")
            playtime = float(row.get("playtime_minutes", 0))
            if not isinstance(provider, str) or not 1 <= len(provider) <= 60 or not 0 <= playtime < 100000000:
                raise ValueError("Invalid library signal")
            metadata = row.get("metadata", "{}")
            if not isinstance(metadata, str) or not isinstance(json.loads(metadata), dict):
                raise ValueError("Invalid library metadata")
            validated_implicit.append((row["game_id"], provider, playtime, metadata))
        if len({row[0] for row in validated_feedback}) != len(validated_feedback):
            raise ValueError("Conflicting duplicate feedback; resolve before import")
        if len({(row[0], row[1]) for row in validated_implicit}) != len(validated_implicit):
            raise ValueError("Duplicate library signal; resolve before import")
        evaluations = payload.get("personal_evaluations", [])
        outcomes = payload.get("outcomes", [])
        if not isinstance(evaluations, list) or not isinstance(outcomes, list):
            raise ValueError("Invalid evaluation history")
        validated_evaluations, validated_outcomes = [], []
        for row in evaluations:
            if not isinstance(row, dict) or not isinstance(row.get("report"), dict):
                raise ValueError("Invalid evaluation report")
            created = float(row.get("created", time.time()))
            if not 0 <= created < 1e12:
                raise ValueError("Invalid evaluation timestamp")
            validated_evaluations.append((json.dumps(row["report"]), created))
        for row in outcomes:
            if not isinstance(row, dict) or row.get("game_id") not in game_ids:
                raise ValueError("Outcome references game absent from export")
            if row.get("kind") not in ("recommendation_relevance", "post_play_enjoyment") or type(row.get("value")) is not int or row["value"] not in RATING_LABELS:
                raise ValueError("Invalid recommendation outcome")
            created = float(row.get("created", time.time()))
            if not 0 <= created < 1e12:
                raise ValueError("Invalid outcome timestamp")
            validated_outcomes.append((row["game_id"], row["kind"], row["value"], created))
        profile_id, now = uuid.uuid4().hex, time.time()
        with self.connection() as db:
            db.execute("INSERT INTO profiles VALUES (?,?,?)", (profile_id, name, now))
            db.executemany("INSERT OR IGNORE INTO games VALUES (?,?,?,?,?,?,?)", validated_games)
            db.executemany("INSERT INTO feedback VALUES (?,?,?,?,?,?,?)", [(profile_id, *row, now) for row in validated_feedback])
            db.executemany("INSERT INTO implicit_signals VALUES (?,?,?,?,?,?)", [(profile_id, *row, now) for row in validated_implicit])
            db.executemany("INSERT INTO personal_evaluations VALUES (?,?,?)", [(profile_id, *row) for row in validated_evaluations])
            db.executemany("INSERT INTO outcomes VALUES (?,?,?,?,?)", [(profile_id, *row) for row in validated_outcomes])
        return profile_id
