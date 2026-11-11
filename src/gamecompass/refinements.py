"""Small, evidence-grounded recommendation filters, independent of ratings.

A Steam listing date does not establish an original work's release date. Unknown
facts stay unknown and cannot satisfy a requested date or edition-kind filter.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Iterable, Literal

# Exact recognized genre labels only. "Story Rich", "Singleplayer", visual
# styles, moods, and popularity tags are deliberately not promoted to genres.
GENRE_TAGS = {
    label.casefold(): label for label in (
        "Action", "Adventure", "RPG", "JRPG", "CRPG", "Action RPG", "Tactical RPG",
        "Strategy", "Turn-Based Strategy", "RTS", "Simulation", "Sports", "Racing",
        "Casual", "Puzzle", "Platformer", "3D Platformer", "2D Platformer", "Shooter",
        "FPS", "Third-Person Shooter", "Fighting", "Beat 'em up", "Survival", "Horror",
        "Survival Horror", "Stealth", "Visual Novel", "Rhythm", "Roguelike", "Roguelite",
        "Metroidvania", "MMORPG", "Tower Defense", "Card Game", "Board Game",
        "Point & Click", "Interactive Fiction", "Dungeon Crawler", "Hack and Slash",
    )
}
PLATFORM_ALIASES = {
    "microsoft windows": "Windows", "windows": "Windows", "pc (microsoft windows)": "Windows",
    "mac": "macOS", "macos": "macOS", "os x": "macOS", "linux": "Linux",
    "pc": "PC", "ps5": "PlayStation 5", "playstation 5": "PlayStation 5",
    "ps4": "PlayStation 4", "playstation 4": "PlayStation 4",
    "ps3": "PlayStation 3", "playstation 3": "PlayStation 3",
    "ps2": "PlayStation 2", "playstation 2": "PlayStation 2",
    "ps1": "PlayStation", "playstation": "PlayStation",
    "switch": "Nintendo Switch", "nintendo switch": "Nintendo Switch",
    "xbox series x and series s": "Xbox Series X|S", "xbox series x|s": "Xbox Series X|S", "xbox series x/s": "Xbox Series X|S",
    "xbox one": "Xbox One", "xbox 360": "Xbox 360",
}
RELEASE_KINDS = ("original", "remake", "remaster", "rerelease", "unknown")


def normalize_platform(value: str) -> str:
    cleaned = value.strip()
    return PLATFORM_ALIASES.get(cleaned.casefold(), cleaned)


def verified_platforms(game: dict[str, Any]) -> list[str]:
    """Catalog/manual platform assertions only; never infer console availability."""
    values = game.get("platforms", [])
    if not isinstance(values, list):
        return []
    return sorted({normalize_platform(value) for value in values if isinstance(value, str) and value.strip()})


def genre_facts(game: dict[str, Any]) -> list[dict[str, str]]:
    facts: dict[str, dict[str, str]] = {}
    genres = game.get("genres", [])
    if isinstance(genres, list):
        for value in genres:
            if isinstance(value, str) and value.strip():
                label = GENRE_TAGS.get(value.strip().casefold(), value.strip())
                facts[label.casefold()] = {"genre": label, "source": "catalog"}
    metadata = game.get("metadata", {})
    tags = metadata.get("tags", []) if isinstance(metadata, dict) else []
    if isinstance(tags, list):
        for tag in tags:
            label = GENRE_TAGS.get(tag.strip().casefold()) if isinstance(tag, str) else None
            if label and label.casefold() not in facts:
                facts[label.casefold()] = {"genre": label, "source": "catalog tag (inferred)"}
    return [facts[key] for key in sorted(facts)]


def available_genres(games: Iterable[dict[str, Any]]) -> list[str]:
    return sorted({fact["genre"] for game in games for fact in genre_facts(game)}, key=str.casefold)


def available_platforms(games: Iterable[dict[str, Any]]) -> list[str]:
    return sorted({platform for game in games for platform in verified_platforms(game)}, key=str.casefold)


def _date(value: Any) -> str | None:
    """Normalize explicit dates/years, without interpreting a store date as origin."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        if 1950 <= value <= 2200 and int(value) == value:
            return f"{int(value):04d}"
        try:
            result = datetime.fromtimestamp(value, tz=timezone.utc).date()
        except (ValueError, OverflowError, OSError):
            return None
        return result.isoformat() if 1950 <= result.year <= 2200 else None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if len(text) == 4 and text.isdigit() and 1950 <= int(text) <= 2200:
        return text
    if len(text)==7 and text[:4].isdigit() and text[4]=='-' and text[5:].isdigit():
        year,month=int(text[:4]),int(text[5:])
        return text if 1950<=year<=2200 and 1<=month<=12 else None
    try:
        result = date.fromisoformat(text[:10]) if len(text) >= 10 else None
    except ValueError:
        result = None
    if result is not None and 1950 <= result.year <= 2200:
        return result.isoformat()
    for pattern in ("%b %d, %Y", "%B %d, %Y", "%d %b, %Y", "%d %B, %Y"):
        try:
            result = datetime.strptime(text, pattern).date()
            return result.isoformat() if 1950 <= result.year <= 2200 else None
        except ValueError:
            pass
    return None


def release_facts(game: dict[str, Any], platform: str | None = None) -> dict[str, Any]:
    """Return dates and edition kind with their evidence; never inspect title text.

    Supported manual format: metadata.manual_release with source=user-provided,
    original_date, edition_date and release_kind. IGDB must have an exact accepted
    mapping; only a main-game record without a parent/version relation establishes
    its own original date. A remake/remaster's first date remains its edition date.
    IGDB remakes/remasters ID lists on an original do not describe that original
    as a remake, and no reverse relationship is invented.
    """
    facts: dict[str, Any] = {"original_date": None, "edition_date": None,
                            "original_year": None, "edition_year": None,
                            "release_kind": "unknown", "original_source": None,
                            "edition_source": None, "kind_source": None}
    metadata = game.get("metadata", {})
    if not isinstance(metadata, dict):
        return facts
    listing_date = _date(metadata.get("date_release") or metadata.get("release_date"))
    if listing_date:
        facts.update(edition_date=listing_date,
                     edition_source=f"{metadata.get('source', 'catalog')} listing/release date")
    wikidata = metadata.get('wikidata', {})
    if (isinstance(wikidata,dict) and game.get('game_id') == 'wikidata:' + str(wikidata.get('entity_id',''))):
        entity_date = _date(metadata.get('entity_release_date'))
        if entity_date:
            facts.update(edition_date=entity_date,edition_source='Wikidata source-entity publication; original work date unknown')
        parent_date=wikidata.get('original_work_release_date')
        if (isinstance(parent_date,dict) and parent_date.get('relation')=='P629'
                and parent_date.get('parent_entity_id') in wikidata.get('version_of',[])
                and parent_date.get('relation_statement') and parent_date.get('publication_statement')
                and parent_date.get('parent_edition_relations') == []
                and any(str(row.get('value','')).endswith('/Q7889') for row in parent_date.get('parent_type_facts',[]))):
            original=_date(parent_date.get('iso'))
            if original:
                facts.update(original_date=original,original_source='Wikidata explicit parent-work relation; earliest observed publication, not proof of ultimate original')
        if platform:
            platform_ids={row.get('value') for row in wikidata.get('platform_facts',[]) if row.get('label')
                          and normalize_platform(row['label'])==normalize_platform(platform)}
            dates=[_date(row.get('iso')) for row in wikidata.get('publication_dates',[])
                   if platform_ids.intersection(row.get('qualifiers',{}).get('P400',[]))]
            dates=[value for value in dates if value]
            facts.update(edition_date=min(dates) if dates else None,
                         edition_source='Wikidata publication qualified for '+platform if dates else 'Platform-specific release date unknown',
                         platform=normalize_platform(platform))
    mapping, igdb = metadata.get("igdb_mapping", {}), metadata.get("igdb", {})
    if (isinstance(mapping, dict) and mapping.get("status") == "matched" and mapping.get("confidence") == 1.0
            and isinstance(igdb, dict) and mapping.get("igdb_id") == igdb.get("id")):
        first_date = _date(igdb.get("first_release_date"))
        game_type = igdb.get("game_type", {})
        type_label = game_type.get("type", "") if isinstance(game_type, dict) else game_type
        type_label = str(type_label).casefold().replace("_", " ").strip()
        kinds = {"main game": "original", "main": "original", "remake": "remake", "remaster": "remaster"}
        kind = kinds.get(type_label, "unknown")
        # A parent/version record may be a special edition; do not promote it to
        # an original merely because a generic game type says main game.
        if kind == "original" and (igdb.get("parent_game") or igdb.get("version_parent")):
            kind = "unknown"
        if first_date:
            facts.update(edition_date=first_date, edition_source="IGDB exact external-ID mapping")
        if kind != "unknown":
            facts.update(release_kind=kind, kind_source="IGDB exact mapping and game type")
        if kind == "original" and first_date:
            facts.update(original_date=first_date, original_source="IGDB exact mapping; main game without parent/version")
    manual = metadata.get("manual_release", {})
    if isinstance(manual, dict) and manual.get("source") == "user-provided":
        for field in ("original", "edition"):
            value = _date(manual.get(f"{field}_date"))
            if value:
                facts[f"{field}_date"] = value
                facts[f"{field}_source"] = "user-provided"
        if manual.get("release_kind") in RELEASE_KINDS:
            facts.update(release_kind=manual["release_kind"], kind_source="user-provided")
    # Explicit legacy manual fields are supported, but a generic release_date
    # from any provider can never fill original_date.
    if metadata.get("source") == "manual":
        original = _date(metadata.get("original_release_date"))
        if original and not facts["original_date"]:
            facts.update(original_date=original, original_source="user-provided")
        kind = metadata.get("release_kind")
        if kind in RELEASE_KINDS and facts["release_kind"] == "unknown":
            facts.update(release_kind=kind, kind_source="user-provided")
    for field in ("original", "edition"):
        value = facts[f"{field}_date"]
        facts[f"{field}_year"] = int(value[:4]) if value else None
    return facts


@dataclass(frozen=True)
class RecommendationFilters:
    platform: str | None = None
    genre: str | None = None
    year_min: int | None = None
    year_max: int | None = None
    era_basis: Literal["original", "edition"] = "original"
    edition_policy: Literal["any", "originals_only", "remakes_remasters_only"] = "any"

    def __post_init__(self) -> None:
        for field in ("platform", "genre"):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, str) or not value.strip() or len(value) > 120):
                raise ValueError(f"{field} must be a nonempty label or null")
        for field in ("year_min", "year_max"):
            value = getattr(self, field)
            if value is not None and (type(value) is not int or not 1950 <= value <= 2200):
                raise ValueError(f"{field} must be an integer from 1950 to 2200")
        if self.year_min is not None and self.year_max is not None and self.year_min > self.year_max:
            raise ValueError("Earliest year must not follow latest year")
        if self.era_basis not in ("original", "edition"):
            raise ValueError("era_basis must be original or edition")
        if self.edition_policy not in ("any", "originals_only", "remakes_remasters_only"):
            raise ValueError("Unsupported edition policy")

    def matches(self, game: dict[str, Any]) -> bool:
        if self.platform and normalize_platform(self.platform) not in verified_platforms(game):
            return False
        if self.genre and self.genre.strip().casefold() not in {fact["genre"].casefold() for fact in genre_facts(game)}:
            return False
        facts = release_facts(game,platform=self.platform if self.era_basis=='edition' else None)
        if self.year_min is not None or self.year_max is not None:
            year = facts[f"{self.era_basis}_year"]
            if year is None or (self.year_min is not None and year < self.year_min) or (self.year_max is not None and year > self.year_max):
                return False
        kind = facts["release_kind"]
        if self.edition_policy == "originals_only" and kind != "original":
            return False
        if self.edition_policy == "remakes_remasters_only" and kind not in ("remake", "remaster"):
            return False
        return True


def catalog_filter_options(store: Any) -> dict[str, list[str]]:
    """Read only facets, in batches, without materializing the whole catalog."""
    import json
    platforms: set[str] = set()
    genres: set[str] = set()
    with store.connection() as connection:
        cursor = connection.execute("SELECT platforms,genres,metadata FROM games")
        while rows := cursor.fetchmany(1000):
            for row in rows:
                try:
                    game = {key: json.loads(row[key]) for key in ("platforms", "genres", "metadata")}
                except (ValueError, TypeError):
                    continue
                platforms.update(verified_platforms(game))
                genres.update(fact["genre"] for fact in genre_facts(game))
    return {"platforms": sorted(platforms, key=str.casefold), "genres": sorted(genres, key=str.casefold)}
