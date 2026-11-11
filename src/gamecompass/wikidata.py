"""CC0 Wikidata entity-scope metadata; never inherit work facts onto Steam editions.

Original code is MIT. Structured Wikidata data is CC0. No media files or article
text are acquired. Queries are bounded, cached, rate limited, and resumable.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import time
import unicodedata
from typing import Any, Callable, Sequence

import requests

from .models import atomic_json, sha256

LOGGER = logging.getLogger(__name__)
ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "GameCompass/0.1 (https://github.com/LPF9000/gamecompass; local noncommercial metadata research)"
PROPERTIES = ("P1733", "P31", "P629", "P747", "P400", "P577", "P136")


class WikidataError(RuntimeError):
    """Bounded acquisition failure; previous successful cache remains reusable."""


def _steam_app(game_id: str) -> str:
    if not isinstance(game_id, str) or not re.fullmatch(r"steam:[1-9][0-9]*", game_id):
        raise ValueError("Wikidata lookup requires canonical Steam IDs")
    return game_id[6:]


def statement_query(game_ids: Sequence[str]) -> str:
    """Exact external-ID relation only; no title search or fuzzy identity merges."""
    if not 1 <= len(game_ids) <= 25:
        raise ValueError("SPARQL batches must contain 1 to 25 Steam IDs")
    values = " ".join('"' + _steam_app(gid) + '"' for gid in game_ids)
    properties = " ".join(f"(wd:{p} p:{p} ps:{p})" for p in PROPERTIES)
    return f'''SELECT ?steam ?item ?itemLabel ?itemDescription ?property ?statement ?value ?valueLabel ?rank ?precision ?qualifierProperty ?qualifierValue ?referenceURL WHERE {{
VALUES ?steam {{ {values} }}
?item wdt:P1733 ?steam .
VALUES (?property ?claimPredicate ?valuePredicate) {{ {properties} }}
?item ?claimPredicate ?statement . ?statement ?valuePredicate ?value; wikibase:rank ?rank .
OPTIONAL {{ ?statement psv:P577 ?timeNode . ?timeNode wikibase:timePrecision ?precision }}
OPTIONAL {{ ?statement ?qualifierProperty ?qualifierValue . FILTER(STRSTARTS(STR(?qualifierProperty), "http://www.wikidata.org/prop/qualifier/")) }}
OPTIONAL {{ ?statement prov:wasDerivedFrom ?reference . ?reference pr:P854 ?referenceURL }}
SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,mul" . }}
}} LIMIT 30000'''


class WikidataClient:
    def __init__(self, *, session: requests.Session | None = None,
                 min_interval: float = 1.0, retries: int = 3,
                 sleep: Callable[[float], None] = time.sleep):
        if min_interval < 1 or not 1 <= retries <= 3:
            raise ValueError("Use at least 1 second between requests and at most 3 attempts")
        self.session = session or requests.Session()
        self.min_interval, self.retries, self.sleep = min_interval, retries, sleep
        self.last_request = 0.0

    def fetch(self, game_ids: Sequence[str]) -> dict:
        query = statement_query(game_ids)
        for attempt in range(self.retries):
            delay = self.min_interval - (time.monotonic() - self.last_request)
            if delay > 0:
                self.sleep(delay)
            self.last_request = time.monotonic()
            try:
                response = self.session.get(ENDPOINT, params={"query": query, "format": "json"},
                    headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"}, timeout=45)
                if response.status_code in (403, 401):
                    raise WikidataError(f"Official Wikidata query endpoint denied access (HTTP {response.status_code}); stopped without alternate-route probing")
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt + 1 < self.retries:
                        retry_after = response.headers.get("Retry-After", "")
                        wait = max(2 ** attempt, float(retry_after)) if re.fullmatch(r"[0-9]+", retry_after) else 2 ** attempt
                        if wait > 60:
                            raise WikidataError("Query service requests a long Retry-After; resume later")
                        self.sleep(wait)
                        continue
                    raise WikidataError(f"Wikidata query failed after bounded retries (HTTP {response.status_code})")
                if response.status_code != 200:
                    raise WikidataError(f"Wikidata query failed (HTTP {response.status_code})")
                payload = response.json()
                bindings = payload.get("results", {}).get("bindings")
                if not isinstance(bindings, list) or any(not isinstance(row, dict) for row in bindings):
                    raise WikidataError("Unexpected SPARQL response schema")
                if len(bindings) >= 30000:
                    raise WikidataError("SPARQL result reached safety limit; split batch before retrying")
                requested = {_steam_app(gid) for gid in game_ids}
                for row in bindings:
                    if row.get("steam", {}).get("value") not in requested:
                        raise WikidataError("SPARQL response includes an unrequested Steam identity")
                return {"endpoint": ENDPOINT, "query": query,
                        "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                        "requested_game_ids": list(game_ids),
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "license": "CC0-1.0", "response": payload}
            except (requests.RequestException, ValueError) as error:
                if attempt + 1 == self.retries:
                    raise WikidataError("Official Wikidata query request failed; cached successful batches preserved") from error
                self.sleep(2 ** attempt)
        raise WikidataError("Wikidata retry limit reached")


def _short(uri: str) -> str:
    return uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]


def _date_iso(value: str, precision: int | None) -> str | None:
    """Preserve known precision; do not fabricate a day for year/month dates."""
    match = re.fullmatch(r"\+?([0-9]{4})-([0-9]{2})-([0-9]{2})T.*", value)
    if not match or precision is None or precision < 9:
        return None
    year, month, day = match.groups()
    if precision == 9:
        return year
    if not 1 <= int(month) <= 12:
        return None
    if precision == 10:
        return f"{year}-{month}"
    try:
        datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")
    except ValueError:
        return None
    return f"{year}-{month}-{day}"


def _title_key(value: str) -> str:
    # Equality gate for text reuse only, never an entity-identity match rule.
    value = unicodedata.normalize("NFKD",value.replace("™","").replace("®","").replace("©", "")).casefold()
    return "".join(char for char in value if char.isalnum() and not unicodedata.combining(char))


def build_catalog(batches: Sequence[dict], steam_catalog: dict[str, dict] | None = None,
                  label_map: dict[str, str] | None = None) -> tuple[list[dict], list[dict]]:
    """Create SEPARATE source-entity records; Steam records are never overwritten.

    P400 applies only to the Wikidata entity that owns the statement. A work,
    compilation or version can each have its own entry. A linked Steam app is a
    relation for provenance, never automatic edition equivalence.
    """
    steam_catalog = steam_catalog or {}
    label_map = label_map or {}
    entities: dict[str, dict] = {}
    mappings: dict[str, set[str]] = defaultdict(set)
    queries: dict[str, set[str]] = defaultdict(set)
    for batch in batches:
        for row in batch["response"]["results"]["bindings"]:
            get = lambda name: row.get(name, {}).get("value")
            qid, app = _short(str(get("item"))), get("steam")
            if not re.fullmatch(r"Q[1-9][0-9]*", qid) or not re.fullmatch(r"[1-9][0-9]*", str(app)):
                raise ValueError("Invalid entity/Steam IDs in SPARQL response")
            mappings[f"steam:{app}"].add(qid)
            queries[qid].add(batch["query_sha256"])
            entity = entities.setdefault(qid, {"label": "", "description": "", "statements": {}, "retrieved_at": set()})
            entity["retrieved_at"].add(batch["retrieved_at"])
            label, description = get("itemLabel"), get("itemDescription")
            if not label or label == qid:
                label = label_map.get(qid)
            if label and label != qid:
                entity["label"] = label
            if description:
                entity["description"] = description
            prop = _short(str(get("property")))
            if prop not in PROPERTIES or not get("statement") or get("value") is None:
                raise ValueError("Invalid statement schema")
            statement = entity["statements"].setdefault(get("statement"), {
                "property": prop, "value": get("value"), "label": None,
                "rank": _short(str(get("rank"))), "precision": None,
                "qualifiers": defaultdict(set), "reference_urls": set()})
            value_id = _short(str(get("value")))
            value_label = get("valueLabel")
            if not value_label or value_label == value_id:
                value_label = label_map.get(value_id)
            if value_label:
                statement["label"] = value_label
            if get("precision"):
                statement["precision"] = int(get("precision"))
            if get("qualifierProperty") and get("qualifierValue"):
                statement["qualifiers"][_short(get("qualifierProperty"))].add(get("qualifierValue"))
            if get("referenceURL"):
                statement["reference_urls"].add(get("referenceURL"))
    reviews = []
    for game_id, qids in sorted(mappings.items()):
        reviews.append({"game_id": game_id, "entity_ids": sorted(qids),
                        "status": "unique_external_relation" if len(qids) == 1 else "review_multiple_entities",
                        "edition_equivalence": "unverified; no cross-entity inheritance"})
    games = []
    for qid, entity in sorted(entities.items()):
        if not entity["label"]:
            reviews.append({"entity_id": qid, "status": "review_missing_label"})
            continue
        facts = []
        for statement_id, statement in sorted(entity["statements"].items()):
            if statement["rank"] == "DeprecatedRank":
                continue
            facts.append({**statement, "statement": statement_id,
                          "qualifiers": {k: sorted(v) for k, v in sorted(statement["qualifiers"].items())},
                          "reference_urls": sorted(statement["reference_urls"])})
        types = [f for f in facts if f["property"] == "P31"]
        # Do not list arbitrary Steam software as a video game. Unknown narrower
        # class ancestry stays reviewable instead of assuming it is a game.
        if not any(_short(f["value"]) == "Q7889" for f in types):
            reviews.append({"entity_id": qid, "status": "review_video_game_type_not_explicit"})
            continue
        platform_facts = [f for f in facts if f["property"] == "P400"]
        platforms = sorted({f["label"] for f in platform_facts if f["label"]})
        genres = sorted({f["label"] for f in facts if f["property"] == "P136" and f["label"]})
        dates = []
        for fact in facts:
            if fact["property"] == "P577":
                dates.append({**fact, "iso": _date_iso(fact["value"], fact["precision"])})
        known_dates = [d for d in dates if d["iso"] is not None]
        earliest = min(known_dates, key=lambda d: (d["iso"], -d["precision"])) if known_dates else None
        related = sorted(gid for gid, qids in mappings.items() if qid in qids)
        all_app_ids = {f"steam:{f['value']}" for f in facts if f["property"] == "P1733"}
        version_of = sorted({_short(f["value"]) for f in facts if f["property"] == "P629"})
        editions = sorted({_short(f["value"]) for f in facts if f["property"] == "P747"})
        scope = "version_or_edition_entity" if version_of else "work_with_editions_entity" if editions else "video_game_entity_scope_not_fully_resolved"
        description = entity["description"]
        desc_source = {"provider": "Wikidata", "field": "entity description"}
        source_tags = []
        tag_source = None
        # A single exact relation supports content reuse, not platform/date
        # inheritance. Multiple app IDs signal broader entity identity.
        if len(related) == 1 and len(all_app_ids) == 1 and len(mappings[related[0]]) == 1 and related[0] in steam_catalog and _title_key(str(steam_catalog[related[0]].get("title", ""))) == _title_key(entity["label"]):
            original = steam_catalog[related[0]]
            original_metadata = original.get("metadata", {})
            if isinstance(original_metadata, str):
                original_metadata = json.loads(original_metadata)
            source_tags = [tag for tag in original_metadata.get("tags", []) if isinstance(tag, str)]
            if source_tags:
                tag_source = {"provider": "Kaggle Steam v28", "game_id": related[0], "reason": "unique exact P1733 relation; copied source tags, not asserted curated Wikidata genres"}
            if original.get("description"):
                description = original["description"]
                desc_source = {"provider": "Kaggle Steam v28", "game_id": related[0],
                               "reason": "unique exact P1733 relation; text reuse does not prove edition equivalence"}
        metadata = {"source": "Wikidata", "source_license": "CC0-1.0", "external_ids": {"wikidata": qid},
            "description_provenance": desc_source, "tags": source_tags, "tags_provenance": tag_source,
            "entity_release_date": earliest["iso"] if earliest else None,
            "entity_release_date_semantics": "earliest observed P577 publication of this source entity; original work date not established",
            "wikidata": {"entity_id": qid, "entity_url": f"https://www.wikidata.org/wiki/{qid}",
                "entity_scope": scope, "related_steam_game_ids": related,
                "all_entity_steam_game_ids": sorted(all_app_ids),
                "identity_status": "external-ID relation only; Steam edition equivalence unverified",
                "mapping_confidence": "exact relation observed; no calibrated probability",
                "type_facts": types, "version_of": version_of, "has_editions": editions,
                "platform_facts": platform_facts, "publication_dates": dates,
                "earliest_observed_publication": {"iso": earliest["iso"], "precision": earliest["precision"]} if earliest else None,
                "original_work_release_date": None,
                "query_sha256": sorted(queries[qid]), "retrieved_at": sorted(entity["retrieved_at"]),
                "evidence_limits": "community-maintained assertions; direct source-entity facts, not independently verified real-world availability"}}
        games.append({"game_id": f"wikidata:{qid}", "title": entity["label"], "description": description,
                      "platforms": platforms, "genres": genres, "themes": [], "metadata": metadata})
    return games, reviews


def complete_labels(batches: Sequence[dict], cache: Path, client: WikidataClient) -> dict:
    """Resolve only missing English/default labels; no guessed platform names."""
    known: dict[str,str] = {}
    missing: set[str] = set()
    for batch in batches:
        for row in batch["response"]["results"]["bindings"]:
            for field,label_field in (("item","itemLabel"),("value","valueLabel")):
                uri = row.get(field,{}).get("value", "")
                qid = _short(uri)
                if not re.fullmatch(r"Q[1-9][0-9]*",qid):
                    continue
                label = row.get(label_field,{}).get("value", "")
                if label and label != qid:
                    known[qid] = label
                else:
                    missing.add(qid)
    missing = sorted(missing-set(known))
    hashes = []
    for offset in range(0,len(missing),100):
        group = missing[offset:offset+100]
        values = " ".join("wd:"+qid for qid in group)
        query = f'''SELECT ?item ?label WHERE {{ VALUES ?item {{ {values} }}
OPTIONAL {{ ?item rdfs:label ?english FILTER(LANG(?english)="en") }}
OPTIONAL {{ ?item rdfs:label ?default FILTER(LANG(?default)="mul") }}
BIND(COALESCE(?english, ?default) AS ?label) }}'''
        digest = hashlib.sha256(query.encode()).hexdigest()
        path = cache / "labels" / f"{digest}.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result.get("query_sha256") != digest:
                raise ValueError("Invalid cached label provenance")
        else:
            payload = None
            for attempt in range(client.retries):
                delay = client.min_interval - (time.monotonic()-client.last_request)
                if delay>0:
                    client.sleep(delay)
                client.last_request = time.monotonic()
                try:
                    response=client.session.get(ENDPOINT,params={"query":query,"format":"json"},headers={"User-Agent":USER_AGENT,"Accept":"application/sparql-results+json"},timeout=45)
                    if response.status_code in (401,403):
                        raise WikidataError(f"Official Wikidata label query denied (HTTP {response.status_code})")
                    if response.status_code==429 or response.status_code>=500:
                        if attempt+1==client.retries:
                            raise WikidataError(f"Wikidata label query failed after bounded retries (HTTP {response.status_code})")
                        retry_after=response.headers.get("Retry-After","")
                        wait=max(2**attempt,float(retry_after)) if re.fullmatch(r"[0-9]+",retry_after) else 2**attempt
                        if wait>60:
                            raise WikidataError("Label query requests long Retry-After; resume later")
                        client.sleep(wait)
                        continue
                    if response.status_code!=200:
                        raise WikidataError(f"Wikidata label query failed (HTTP {response.status_code})")
                    payload=response.json()
                    if not isinstance(payload.get("results",{}).get("bindings"),list):
                        raise WikidataError("Invalid label query response schema")
                    break
                except (requests.RequestException,ValueError) as error:
                    if attempt+1==client.retries:
                        raise WikidataError("Wikidata label request failed; resume cached acquisition later") from error
                    client.sleep(2**attempt)
            result={"query_sha256":digest,"query":query,"response":payload,"retrieved_at":datetime.now(timezone.utc).isoformat(),"license":"CC0-1.0"}
            atomic_json(path,result)
        hashes.append(digest)
        for row in result["response"]["results"]["bindings"]:
            qid=_short(row.get("item",{}).get("value",""))
            label=row.get("label",{}).get("value")
            if qid not in group:
                raise WikidataError("Label query returned unrequested identity")
            if label:
                known[qid]=label
    return {"labels":known,"query_sha256":hashes}


def acquire_enrichment(data_dir: Path, game_ids: Sequence[str], *, client: WikidataClient | None = None,
                       apply: bool = False) -> dict[str, Any]:
    """Cache bounded batches, produce compact public catalog, optionally update local games only."""
    data_dir = Path(data_dir)
    ids = list(dict.fromkeys(game_ids))
    if not 1 <= len(ids) <= 1500:
        raise ValueError("Request 1 to 1500 canonical Steam IDs")
    for gid in ids:
        _steam_app(gid)
    cache = data_dir / "wikidata"
    cache.mkdir(parents=True, exist_ok=True)
    client = client or WikidataClient()
    source = data_dir / "population.sqlite"
    original_hash = sha256(source) if source.exists() else None
    batches, errors = [], []
    started = time.monotonic()
    for offset in range(0, len(ids), 25):
        group = ids[offset:offset+25]
        query = statement_query(group)
        digest = hashlib.sha256(query.encode()).hexdigest()
        path = cache / "batches" / f"{digest}.json"
        # Earlier successful en-only batches contain valid statements; reuse them
        # and resolve missing mul labels separately rather than repeat all data queries.
        legacy_query = query.replace('wikibase:language "en,mul"', 'wikibase:language "en"')
        legacy_digest = hashlib.sha256(legacy_query.encode()).hexdigest()
        legacy_path = cache / "batches" / f"{legacy_digest}.json"
        if not path.exists() and legacy_path.exists():
            path, digest = legacy_path, legacy_digest
        if path.exists():
            batch = json.loads(path.read_text())
            if batch.get("query_sha256") != digest or batch.get("requested_game_ids") != group:
                raise ValueError("Invalid Wikidata cache provenance")
        else:
            try:
                batch = client.fetch(group)
            except WikidataError as error:
                errors.append({"batch_offset": offset, "error": str(error)})
                break  # Do not mass-query a blocked/unavailable public service.
            atomic_json(path, batch)
        batches.append(batch)
        LOGGER.info("Wikidata: %d/%d Steam IDs queried; %d result rows", min(offset+25,len(ids)),len(ids),len(batch["response"]["results"]["bindings"]))
    # Read permitted source catalog without changing population/user records.
    import sqlite3
    steam_catalog = {}
    requested_ids = set(ids)
    if source.exists():
        with sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            for row in db.execute("SELECT * FROM games"):
                if row["game_id"] in requested_ids:
                    steam_catalog[row["game_id"]] = dict(row)
    labels = complete_labels(batches, cache, client)
    games, reviews = build_catalog(batches, steam_catalog, labels["labels"])
    for game in games:
        game["metadata"]["wikidata"]["label_query_sha256"] = labels["query_sha256"]
    atomic_json(cache / "catalog.json", {"schema_version": 1, "provider": "Wikidata", "license": "CC0-1.0", "games": games})
    atomic_json(cache / "mapping-review.json", reviews)
    if apply:
        from .store import Store
        store = Store(data_dir)
        for game in games:
            store.upsert_game(game)
    if source.exists() and sha256(source) != original_hash:
        raise RuntimeError("Population source changed during enrichment")
    platform_counts = Counter(p for game in games for p in game["platforms"])
    report = {"requested_steam_ids": len(ids), "successful_batches": len(batches), "completed": not errors,
        "catalog_entities": len(games), "with_platform_facts": sum(bool(g["platforms"]) for g in games),
        "with_entity_publication_date": sum(bool(g["metadata"]["entity_release_date"]) for g in games),
        "original_work_dates_asserted": 0,
        "steam_description_reuse": sum(g["metadata"]["description_provenance"]["provider"] == "Kaggle Steam v28" for g in games),
        "steam_tag_reuse": sum(bool(g["metadata"].get("tags")) for g in games),
        "platform_counts": dict(platform_counts.most_common()),
        "mapping_status_counts": dict(Counter(r["status"] for r in reviews)),
        "population_sha256_before_after": original_hash, "applied_to_local_catalog": apply,
        "catalog_sha256": sha256(cache/"catalog.json"), "catalog_bytes": (cache/"catalog.json").stat().st_size,
        "errors": errors, "duration_seconds": round(time.monotonic()-started,3),
        "identity_policy": "Separate wikidata:Q source entities only. Do not transfer P400/P577 onto Steam editions. No original-work-date inference from missing relations."}
    atomic_json(cache / "report.json", report)
    return report


def resolve_parent_work_date(child: dict, parent_facts: Sequence[dict], relation_facts: Sequence[dict]) -> dict:
    """Resolve a relative P629 parent-work date, never infer an ultimate original.

    Requires positive child P629 AND reciprocal parent P747 evidence, an explicit
    video-game parent type, no parent P629/series/compilation type, and consistent
    same-context publication dates. Platform/region-qualified ports can differ.
    """
    wd = child.get("metadata", {}).get("wikidata", {})
    parents = wd.get("version_of", [])
    result = {"status": "unknown", "reason": "unverified or multiple explicit parents", "date": None}
    if len(parents) != 1 or child.get("game_id") != "wikidata:" + str(wd.get("entity_id")):
        return result
    parent, child_qid = parents[0], wd["entity_id"]
    direct = [f for f in relation_facts if f.get("property") == "P629" and _short(f.get("value", "")) == parent and f.get("rank") != "DeprecatedRank"]
    facts = [f for f in parent_facts if f.get("rank") != "DeprecatedRank"]
    types = {_short(f.get("value", "")) for f in facts if f.get("property") == "P31"}
    if not direct or "Q7889" not in types or types.intersection({"Q7058673", "Q16070115"}):
        return {**result, "reason": "parent is not an explicitly identified individual video-game work"}
    if any(f.get("property") == "P629" for f in facts):
        return {**result, "reason": "parent is itself a version/edition; no traversal or ultimate-original inference"}
    reciprocal = [f for f in facts if f.get("property") == "P747" and _short(f.get("value", "")) == child_qid]
    if not reciprocal:
        return {**result, "reason": "missing positive reciprocal parent-work/edition evidence"}
    dates, contexts = [], defaultdict(set)
    for fact in facts:
        if fact.get("property") != "P577":
            continue
        iso = _date_iso(fact.get("value", ""), fact.get("precision"))
        if not iso:
            continue
        qualifiers = fact.get("qualifiers", {})
        if set(qualifiers) - {"P400", "P291"}:
            return {**result, "reason": "parent publication has unsupported version/context qualifiers"}
        context = json.dumps(qualifiers, sort_keys=True)
        contexts[context].add(iso)
        dates.append({**fact, "iso": iso})
    if not dates:
        return {**result, "reason": "parent has no usable publication date"}
    if any(len(values) > 1 for values in contexts.values()):
        return {**result, "reason": "contradictory publication dates in the same source context"}
    first = min(dates, key=lambda f: (f["iso"], -f["precision"]))
    return {"status": "resolved_relative_parent_work", "reason": "explicit bidirectional P629/P747 work-edition relation",
            "date": {"iso": first["iso"], "precision": first["precision"], "parent_entity_id": parent,
                "parent_entity_url": f"https://www.wikidata.org/wiki/{parent}",
                "parent_type_facts": [f for f in facts if f.get("property") == "P31"],
                "parent_edition_relations": [f for f in facts if f.get("property") == "P629"],
                "relation": "P629", "relation_statement": direct[0]["statement"],
                "reciprocal_relation_statement": reciprocal[0]["statement"],
                "publication_statement": first["statement"], "publication_qualifiers": first.get("qualifiers", {}),
                "reference_urls": first.get("reference_urls", []),
                "source": "Wikidata explicit P629 parent work; earliest known publication",
                "semantics": "relative parent-work publication, not proof of ultimate original work or a console port date"}}


def enrich_parent_work_dates(data_dir: Path, *, apply: bool = False,
                             client: WikidataClient | None = None) -> dict:
    """Optional one-hop metadata follow-up, bounded to at most 25 actual parents."""
    data_dir = Path(data_dir)
    cache = data_dir / "wikidata"
    catalog = json.loads((cache / "catalog.json").read_text())
    candidates = [g for g in catalog["games"] if len(g["metadata"]["wikidata"].get("version_of", [])) == 1]
    parents = sorted({g["metadata"]["wikidata"]["version_of"][0] for g in candidates})
    if not parents:
        return {"candidate_entries": 0, "resolved_entries": 0, "status": "no explicit P629 parent relations"}
    if len(parents) > 25 or any(not re.fullmatch(r"Q[1-9][0-9]*", qid) for qid in parents):
        raise ValueError("Parent-work follow-up is bounded to 25 valid explicit parents")
    source = data_dir / "population.sqlite"
    source_hash = sha256(source) if source.exists() else None
    path = cache / "parent-work.json"
    client = client or WikidataClient()
    if path.exists():
        parent_batch = json.loads(path.read_text())
        if parent_batch.get("requested_parent_ids") != parents or parent_batch.get("query_sha256") != hashlib.sha256(parent_batch["query"].encode()).hexdigest():
            raise ValueError("Cached parent-work query provenance differs; preserve or refresh explicitly")
    else:
        values = " ".join("wd:" + qid for qid in parents)
        query = f'''SELECT ?item ?itemLabel ?property ?statement ?value ?valueLabel ?rank ?precision ?qualifierProperty ?qualifierValue ?referenceURL WHERE {{
VALUES ?item {{ {values} }}
VALUES (?property ?cp ?vp) {{ (wd:P31 p:P31 ps:P31) (wd:P629 p:P629 ps:P629) (wd:P747 p:P747 ps:P747) (wd:P577 p:P577 ps:P577) }}
?item ?cp ?statement . ?statement ?vp ?value; wikibase:rank ?rank .
OPTIONAL {{?statement psv:P577 ?timeNode . ?timeNode wikibase:timePrecision ?precision}}
OPTIONAL {{?statement ?qualifierProperty ?qualifierValue FILTER(STRSTARTS(STR(?qualifierProperty),"http://www.wikidata.org/prop/qualifier/"))}}
OPTIONAL {{?statement prov:wasDerivedFrom ?reference . ?reference pr:P854 ?referenceURL}}
SERVICE wikibase:label {{bd:serviceParam wikibase:language "en,mul"}}
}} LIMIT 10000'''
        payload = None
        for attempt in range(client.retries):
            delay = client.min_interval - (time.monotonic() - client.last_request)
            if delay > 0:
                client.sleep(delay)
            client.last_request = time.monotonic()
            try:
                response = client.session.get(ENDPOINT, params={"query":query,"format":"json"},
                    headers={"User-Agent":USER_AGENT,"Accept":"application/sparql-results+json"},timeout=45)
                if response.status_code in (401,403):
                    raise WikidataError(f"Parent-work query denied (HTTP {response.status_code})")
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt+1 == client.retries:
                        raise WikidataError("Parent-work query unavailable after bounded retries")
                    retry_after=response.headers.get("Retry-After","")
                    wait=max(2**attempt,float(retry_after)) if re.fullmatch(r"[0-9]+",retry_after) else 2**attempt
                    if wait>60:
                        raise WikidataError("Parent-work query requests long Retry-After; resume later")
                    client.sleep(wait)
                    continue
                if response.status_code != 200:
                    raise WikidataError(f"Parent-work query failed (HTTP {response.status_code})")
                payload=response.json()
                if not isinstance(payload.get("results",{}).get("bindings"),list) or len(payload["results"]["bindings"])>=10000:
                    raise WikidataError("Invalid or truncated parent-work query")
                break
            except (requests.RequestException,ValueError) as error:
                if attempt+1==client.retries:
                    raise WikidataError("Parent-work request failed; resume later") from error
                client.sleep(2**attempt)
        parent_batch={"endpoint":ENDPOINT,"requested_parent_ids":parents,"query":query,
            "query_sha256":hashlib.sha256(query.encode()).hexdigest(),"response":payload,
            "retrieved_at":datetime.now(timezone.utc).isoformat(),"license":"CC0-1.0"}
        atomic_json(path,parent_batch)
    grouped: dict[str,dict] = defaultdict(dict)
    for row in parent_batch["response"]["results"]["bindings"]:
        get = lambda key: row.get(key,{}).get("value")
        qid = _short(str(get("item")))
        if qid not in parents:
            raise WikidataError("Parent query returned an unrequested identity")
        sid = get("statement")
        fact=grouped[qid].setdefault(sid,{"property":_short(str(get("property"))),"value":get("value"),
            "statement":sid,"rank":_short(str(get("rank"))),"precision":int(get("precision")) if get("precision") else None,
            "qualifiers":defaultdict(set),"reference_urls":set()})
        if get("qualifierProperty") and get("qualifierValue"):
            fact["qualifiers"][_short(get("qualifierProperty"))].add(get("qualifierValue"))
        if get("referenceURL"):
            fact["reference_urls"].add(get("referenceURL"))
    facts_by_parent={qid:[{**f,"qualifiers":{p:sorted(v) for p,v in f["qualifiers"].items()},"reference_urls":sorted(f["reference_urls"])} for f in statements.values()] for qid,statements in grouped.items()}
    relation_by_child: dict[str,list] = defaultdict(list)
    for batch_path in (cache/"batches").glob("*.json"):
        batch=json.loads(batch_path.read_text())
        for row in batch["response"]["results"]["bindings"]:
            if _short(row.get("property",{}).get("value",""))=="P629":
                child=_short(row["item"]["value"])
                relation_by_child[child].append({"property":"P629","value":row["value"]["value"],
                    "statement":row["statement"]["value"],"rank":_short(row["rank"]["value"])})
    outcomes=[]
    for child in candidates:
        wd=child["metadata"]["wikidata"]
        decision=resolve_parent_work_date(child,facts_by_parent.get(wd["version_of"][0],[]),relation_by_child.get(wd["entity_id"],[]))
        wd["parent_work_date_resolution"]={"status":decision["status"],"reason":decision["reason"],"query_sha256":parent_batch["query_sha256"],"retrieved_at":parent_batch["retrieved_at"]}
        if decision["date"] is not None:
            wd["original_work_release_date"]={**decision["date"],"query_sha256":parent_batch["query_sha256"],"retrieved_at":parent_batch["retrieved_at"]}
        outcomes.append({"game_id":child["game_id"],"title":child["title"],**decision})
    if apply:
        from .store import Store
        store=Store(data_dir)
        for child in candidates:
            store.upsert_game(child)
    if source.exists() and sha256(source)!=source_hash:
        raise RuntimeError("Population source changed during parent-work enrichment")
    atomic_json(cache/"catalog.json",catalog)
    report={"candidate_entries":len(candidates),"queried_parent_entities":len(parents),
        "resolved_entries":sum(row["date"] is not None for row in outcomes),"outcomes":outcomes,
        "source_population_unchanged_sha256":source_hash,"applied_to_local_catalog":apply,
        "catalog_bytes":(cache/"catalog.json").stat().st_size,"catalog_sha256":sha256(cache/"catalog.json"),
        "semantics":"Explicit P629 relative-parent work date, not ultimate-original inference"}
    atomic_json(cache/"parent-work-report.json",report)
    aggregate_path = cache / "report.json"
    if aggregate_path.exists():
        aggregate = json.loads(aggregate_path.read_text())
        aggregate.update({"original_work_dates_asserted": report["resolved_entries"],
            "relative_parent_work_dates_resolved": report["resolved_entries"],
            "original_work_date_semantics": "relative explicit P629 parent work only; ultimate-original unestablished",
            "catalog_bytes": report["catalog_bytes"], "catalog_sha256": report["catalog_sha256"]})
        atomic_json(aggregate_path, aggregate)
    return report
