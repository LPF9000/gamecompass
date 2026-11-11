"""Optional authorized IGDB enrichment; deterministic identities, no fuzzy merges.

No network requests or caching happen on import. Credentials stay in memory.
Cached provider assets remain subject to their own terms, never the code license.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any

import requests

from gamecompass.store import Store


class CatalogError(RuntimeError):
    """A credential-safe provider failure."""


@dataclass(frozen=True)
class MappingDecision:
    status: str
    igdb_id: int | None
    confidence: float
    reason: str
    candidates: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _steam_id(source: dict[str, Any]) -> str:
    game_id = str(source.get("game_id", ""))
    if not re.fullmatch(r"steam:[1-9][0-9]*", game_id):
        raise ValueError("Enrichment requires a canonical steam:<positive app ID>")
    return game_id.split(":", 1)[1]


def _external_matches(external: dict[str, Any], app_id: str) -> bool:
    provider = external.get("external_game_source")
    modern_steam = isinstance(provider, dict) and str(provider.get("name", "")).casefold() == "steam"
    # Legacy category=1 is explicitly documented as Steam. Do not assume that
    # the new source reference has the same numeric ID as the deprecated enum.
    legacy_steam = provider is None and external.get("category") == 1
    return str(external.get("uid", "")) == app_id and (modern_steam or legacy_steam)


def resolve_candidates(source: dict[str, Any], candidates: list[dict[str, Any]]) -> MappingDecision:
    """Accept exactly one authoritative external-ID match, otherwise retain review.

    Confidence is a deterministic evidence label, not a calibrated probability.
    Candidate shape is IGDB game objects with expanded external_games records.
    Similar or identical names, franchises, and parent IDs never imply identity.
    """
    app_id = _steam_id(source)
    valid = [candidate for candidate in candidates if isinstance(candidate, dict)
             and type(candidate.get("id")) is int and candidate["id"] > 0]
    summaries: dict[int, dict[str, Any]] = {}
    matches: dict[int, dict[str, Any]] = {}
    for candidate in valid:
        external_records = candidate.get("external_games") or []
        if not isinstance(external_records, list):
            external_records = []
        exact = any(isinstance(external, dict) and _external_matches(external, app_id)
                    for external in external_records)
        summary = {"id": candidate["id"], "name": str(candidate.get("name", "")),
                   "confidence": 1.0 if exact else 0.0,
                   "evidence": "exact Steam app ID" if exact else "no verified Steam app ID"}
        if candidate["id"] not in summaries or exact:
            summaries[candidate["id"]] = summary
        if exact:
            matches[candidate["id"]] = candidate
    retained = tuple(summaries[key] for key in sorted(summaries))
    if len(matches) == 1:
        candidate = next(iter(matches.values()))
        source_kind = (source.get("metadata") or {}).get("game_type")
        candidate_kind = candidate.get("game_type")
        if isinstance(candidate_kind, dict):
            candidate_kind = candidate_kind.get("type", candidate_kind.get("name"))
        if source_kind and candidate_kind and str(source_kind).casefold() != str(candidate_kind).casefold():
            return MappingDecision("review", None, 0.0, "External ID matches but game types conflict", retained)
        return MappingDecision("matched", candidate["id"], 1.0, "Unique exact Steam app ID", retained)
    if len(matches) > 1:
        return MappingDecision("review", None, 0.0, "Steam app ID maps to multiple IGDB games", retained)
    return MappingDecision("review" if retained else "unmatched", None, 0.0,
                           "No verified external ID; names are never merged automatically", retained)


class IGDBClient:
    """Synchronous bounded API client; caller runs lengthy work off the UI thread."""

    def __init__(self, *, client_id: str | None = None, access_token: str | None = None,
                 session: requests.Session | None = None):
        self._client_id = client_id or os.getenv("IGDB_CLIENT_ID", "")
        self._access_token = access_token or os.getenv("IGDB_ACCESS_TOKEN", "")
        if not self._client_id or not self._access_token:
            raise CatalogError("Set IGDB_CLIENT_ID and IGDB_ACCESS_TOKEN for authorized enrichment")
        if any("\n" in value or "\r" in value for value in (self._client_id, self._access_token)):
            raise CatalogError("Invalid IGDB credentials")
        self._session = session or requests.Session()
        self._last_request = 0.0

    def _query(self, endpoint: str, query: str) -> list[dict[str, Any]]:
        if endpoint not in {"external_game_sources", "external_games", "games"}:
            raise ValueError("Unsupported catalog endpoint")
        for attempt in range(3):
            delay = .26 - (time.monotonic() - self._last_request)
            if delay > 0:
                time.sleep(delay)
            self._last_request = time.monotonic()
            try:
                response = self._session.post(f"https://api.igdb.com/v4/{endpoint}", data=query,
                                              headers={"Client-ID": self._client_id,
                                                       "Authorization": f"Bearer {self._access_token}",
                                                       "Accept": "application/json"}, timeout=30)
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < 2:
                        time.sleep(.5 * 2 ** attempt)
                        continue
                if response.status_code != 200:
                    raise CatalogError(f"IGDB request failed (HTTP {response.status_code})")
                payload = response.json()
                if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
                    raise CatalogError("Invalid IGDB response schema")
                return payload
            except (requests.RequestException, ValueError):
                if attempt == 2:
                    raise CatalogError("IGDB request failed; check connectivity and authorization") from None
                time.sleep(.5 * 2 ** attempt)
        raise CatalogError("IGDB retry limit reached")

    def steam_candidates(self, app_id: str) -> list[dict[str, Any]]:
        if not re.fullmatch(r"[1-9][0-9]*", str(app_id)):
            raise ValueError("Steam app ID must be a positive integer")
        # Discover the new table's ID by exact name rather than reusing enum=1.
        sources = self._query("external_game_sources", 'fields name; where name = "Steam"; limit 10;')
        source_ids = {row["id"] for row in sources if type(row.get("id")) is int
                      and str(row.get("name", "")).casefold() == "steam"}
        if len(source_ids) != 1:
            raise CatalogError("Could not uniquely resolve IGDB's Steam external source")
        source_id = next(iter(source_ids))
        links = self._query("external_games", f'fields uid,game,external_game_source; '
                            f'where external_game_source = {source_id} & uid = "{app_id}"; limit 500;')
        if len(links) >= 500:
            raise CatalogError("IGDB mapping response was truncated; manual review required")
        ids = sorted({row["game"] for row in links if type(row.get("game")) is int
                      and row.get("external_game_source") == source_id and str(row.get("uid")) == str(app_id)})
        if not ids:
            return []
        fields = ("name,alternative_names.name,summary,storyline,first_release_date,platforms.name,genres.name,"
                  "themes.name,game_modes.name,involved_companies.company.name,involved_companies.developer,"
                  "involved_companies.publisher,franchises.name,cover.url,external_games.uid,"
                  "external_games.external_game_source.name,game_type.type,parent_game,remakes,remasters,"
                  "expansions,dlcs,version_parent")
        return self._query("games", f'fields {fields}; where id = ({",".join(map(str, ids))}); limit 500;')


def enrich_steam_game(store: Store, game_id: str, client: IGDBClient, *,
                      acknowledge_provider_terms: bool = False) -> MappingDecision:
    """Opt-in local cache. No automatic name changes or cross-source game merges."""
    if not acknowledge_provider_terms:
        raise CatalogError("Local caching requires explicit acknowledgement of IGDB/Twitch terms")
    source = store.game(game_id)
    if source is None:
        raise ValueError("Add the game to the catalog before enriching it")
    candidates = client.steam_candidates(_steam_id(source))
    decision = resolve_candidates(source, candidates)
    metadata = dict(source.get("metadata", {}))
    metadata["igdb_mapping"] = decision.to_dict()
    metadata["igdb_terms"] = {"acknowledged_at": time.time(), "redistribution_authorized": False}
    updated = dict(source)
    if decision.status == "matched":
        record = next(row for row in candidates if row["id"] == decision.igdb_id)
        metadata["igdb"] = record
        metadata["igdb_retrieved_at"] = time.time()
        updated["description"] = str(record.get("summary") or source.get("description", ""))
        for field in ("platforms", "genres", "themes"):
            names = [row["name"] for row in (record.get(field) or []) if isinstance(row, dict)
                     and isinstance(row.get("name"), str)]
            updated[field] = sorted(set(source.get(field, []) + names))
    updated["metadata"] = metadata
    store.upsert_game(updated)
    return decision
