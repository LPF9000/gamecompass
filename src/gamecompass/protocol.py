"""Prospectively locked, previously unscored whole-user evaluation cohorts.

This module selects identities and freezes provenance; it never scores models.
Raw IDs and histories stay beneath ignored data/. Public summaries are aggregate.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Mapping, Sequence

from .evaluation import HISTORY_LENGTHS, cold_item_partition
from .models import sha256

PROTOCOL_SEED = 20261110
PROTOCOL_VERSION = 2
Row = Sequence[Any]


def _eligible(rows: Sequence[Row], history: int) -> bool:
    return len(rows) > history and any(row[2] for row in rows[history:])


def _cold_item_eligible(rows: Sequence[Row], cold_ids: set[str], history: int) -> bool:
    warm = [i for i, row in enumerate(rows) if row[1] not in cold_ids]
    return len(warm) >= history and any(row[1] in cold_ids and row[2] for row in rows[warm[history - 1] + 1:])


def legacy_exposed_user_ids(test_users: Mapping[str, Sequence[Row]], cf_ids: Sequence[str], *,
                            max_users: int = 300, split_seed: int = 42,
                            additional_ids: Sequence[str] = ()) -> set[str]:
    """Reconstruct the union of *all* earlier primary-dataset test identities.

    Native, implicit-only and semantic cohorts use the same positive-remainder
    eligibility. Cold-item queries use a different warm-prefix eligibility.
    Entire users are excluded even if only one of their queries was inspected.
    Early UCSD runs are another user namespace and never justify a 500-user
    exclusion on this primary Kaggle split. Additional exposures can be supplied.
    """
    if max_users < 1:
        raise ValueError("Legacy exposure cap must be positive")
    sorted_users = sorted(test_users)
    exposed = set(additional_ids) & set(test_users)
    for history in HISTORY_LENGTHS:
        cohort = [user for user in sorted_users if _eligible(test_users[user], history)]
        exposed.update(cohort[:max_users])
    cold_ids = {game for game in cf_ids if cold_item_partition(game, split_seed)}
    for history in (1, 3, 5):
        cohort = [user for user in sorted_users if _cold_item_eligible(test_users[user], cold_ids, history)]
        exposed.update(cohort[:max_users])
    return exposed


def temporal_exposed_user_ids(data_dir: Path, dataset_sha256: str) -> set[str]:
    """Conservatively reconstruct an already-published primary temporal cohort.

    Temporal evaluation can overlap the original whole-user test partition.
    This reconstructs eligibility, never model predictions. Other source hashes
    (including UCSD) are deliberately ignored.
    """
    report_path = data_dir / "temporal-benchmark.json"
    if not report_path.exists():
        return set()
    report = json.loads(report_path.read_text())
    if report.get("dataset_sha256") != dataset_sha256:
        return set()
    cutoff, max_items, cap = report["cutoff_timestamp"], report["max_items"], report["max_users"]
    with sqlite3.connect(f"file:{(data_dir / 'population.sqlite').resolve()}?mode=ro", uri=True) as db:
        available = {row[0] for row in db.execute("SELECT game_id FROM games")}
        counts = Counter({game: count for game, count in db.execute(
            "SELECT game_id,COUNT(*) FROM interactions WHERE timestamp > 0 AND timestamp <= ? AND positive=1 GROUP BY game_id", (cutoff,)) if game in available})
        ids = {game for game, _ in sorted(counts.items(), key=lambda row: (-row[1], row[0]))[:max_items]}
        prior: dict[str, set[str]] = defaultdict(set)
        future: dict[str, set[str]] = defaultdict(set)
        for user, game, positive, timestamp in db.execute(
                "SELECT user_id,game_id,positive,timestamp FROM interactions WHERE timestamp > 0 ORDER BY user_id,timestamp,game_id"):
            if game not in ids:
                continue
            if timestamp <= cutoff:
                prior[user].add(game)
            elif positive:
                future[user].add(game)
    selected = [user for user in sorted(future) if future[user] - prior[user]][:cap]
    if len(selected) != report["evaluation_queries"]:
        raise ValueError("Could not reproduce legacy temporal exposure count")
    return set(selected)


def _hash_order(user: str, seed: int) -> tuple[str, str]:
    return hashlib.sha256(f"fresh-user-v2:{seed}:{user}".encode()).hexdigest(), user


def select_cohort(test_users: Mapping[str, Sequence[Row]], exposed: set[str], history: int, *,
                  cap: int = 500, seed: int = PROTOCOL_SEED) -> list[str]:
    """Deterministic hash sample, selected independently of model scores."""
    if type(history) is not int or history < 0 or cap < 1:
        raise ValueError("History must be nonnegative and cap positive")
    eligible = (user for user, rows in test_users.items() if user not in exposed and _eligible(rows, history))
    return sorted(eligible, key=lambda user: _hash_order(user, seed))[:cap]


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _write_once(path: Path, payload: dict) -> dict:
    """Exclusive creation prevents accidental cohort/config replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical(payload)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())
        # Hard-link publication is atomic and fails if the final name exists.
        # An interrupted write can leave only an irrelevant temporary file,
        # never a partially published protocol or config.
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = json.loads(path.read_text())
            if _canonical(existing) != encoded:
                raise ValueError("Immutable protocol artifact already exists with different content") from None
            return existing
    finally:
        Path(temporary).unlink(missing_ok=True)
    return payload


def create_protocol(data_dir: Path | str, *, base_version: str | None = None,
                    cap: int = 500, seed: int = PROTOCOL_SEED,
                    filename: str = "fresh-users-v2.json") -> dict:
    """Lock the base split and unused test-user identities without scoring.

    Population fit remains the original seed-42 artifact. This is a fresh *user
    sample* from that split, not a new training split or a retraining request.
    """
    data_dir = Path(data_dir)
    if cap < 1 or cap > 500:
        raise ValueError("Prospective final test cap must be between 1 and 500")
    if Path(filename).name != filename:
        raise ValueError("Protocol filename must not contain directories")
    base_version = base_version or json.loads((data_dir / "models" / "current.json").read_text())["version"]
    artifact = data_dir / "models" / base_version
    model_manifest = json.loads((artifact / "manifest.json").read_text())
    split_path = data_dir / "experiments" / base_version / "split.json"
    source_hash = sha256(data_dir / "population.sqlite")
    split_hash = sha256(split_path)
    if source_hash != model_manifest["source_sha256"] or split_hash != model_manifest["split_sha256"]:
        raise ValueError("Frozen population or original split differs from the base artifact")
    prepared = json.loads(split_path.read_text())
    if prepared["seed"] != 42:
        raise ValueError("This legacy exposure reconstruction expects original seed 42")
    users = prepared["test_users"]
    temporal = temporal_exposed_user_ids(data_dir, source_hash)
    exposed = legacy_exposed_user_ids(users, prepared["cf_ids"], additional_ids=sorted(temporal))
    histories = list(HISTORY_LENGTHS)
    cohorts = {str(history): select_cohort(users, exposed, history, cap=cap, seed=seed) for history in histories}
    available = {str(history): sum(user not in exposed and _eligible(rows, history) for user, rows in users.items()) for history in histories}
    checksums = {name: sha256(artifact / name) for name in ("weights.npz", "knn.npz", "content.npz")}
    if checksums != model_manifest["checksums"]:
        raise ValueError("Base model weight checksums do not match")
    manifest = {
        "protocol_version": PROTOCOL_VERSION, "seed": seed, "cap_per_history": cap,
        "base_split_seed": prepared["seed"], "dataset_sha256": source_hash,
        "base_version": base_version, "base_split_sha256": split_hash,
        "base_weight_checksums": checksums, "cf_ids_sha256": hashlib.sha256(_canonical(prepared["cf_ids"])).hexdigest(),
        "legacy_primary_cap": 300, "legacy_exposed_user_ids": sorted(exposed),
        "legacy_temporal_overlap_count": len(temporal & set(users)),
        "original_test_users": len(users), "remaining_unexposed_test_users": len(users) - len(exposed),
        "available_by_history": available, "cohorts": cohorts,
        "selection": "ascending SHA256(fresh-user-v2:seed:user_id), positive remainder after chronological eligible-item prefix",
        "candidate_strategy": "all original training-selected CF items excluding support; no sampled negatives",
        "model_selection": "forbidden on these cohorts; validation users only until all configurations frozen",
        "scope": "new unscored users in fixed bounded historical-review test split; not an independent new dataset",
    }
    return _write_once(data_dir / "protocols" / filename, manifest)


def public_summary(manifest: dict) -> dict:
    """No IDs, histories, scores or preferences enter the public summary."""
    return {key: manifest[key] for key in (
        "protocol_version", "seed", "cap_per_history", "base_split_seed", "dataset_sha256", "base_version",
        "base_split_sha256", "base_weight_checksums", "cf_ids_sha256", "legacy_primary_cap",
        "legacy_temporal_overlap_count", "original_test_users", "remaining_unexposed_test_users",
        "available_by_history", "selection", "candidate_strategy", "model_selection", "scope")} | {
        "protocol_sha256": hashlib.sha256(_canonical(manifest)).hexdigest(),
        "legacy_exposed_users": len(manifest["legacy_exposed_user_ids"]),
        "selected_by_history": {history: len(users) for history, users in manifest["cohorts"].items()},
    }


def cohort_users(manifest: dict, prepared: Mapping, history: int) -> dict[str, Sequence[Row]]:
    """Return frozen users in hash order for existing cold/implicit query APIs.

    Those APIs sort IDs internally: pass max_users >= cohort size so they cannot
    silently cap this already sampled cohort again.
    """
    key = str(history)
    if key not in manifest["cohorts"]:
        raise ValueError("History is absent from the immutable protocol")
    if hashlib.sha256(_canonical(prepared["cf_ids"])).hexdigest() != manifest["cf_ids_sha256"]:
        raise ValueError("Candidate universe differs from the locked protocol")
    exposed = set(manifest["legacy_exposed_user_ids"])
    result = {}
    for user in manifest["cohorts"][key]:
        if user in exposed or user not in prepared["test_users"] or not _eligible(prepared["test_users"][user], history):
            raise ValueError("Frozen cohort eligibility or exposure guard failed")
        result[user] = prepared["test_users"][user]
    return result


def freeze_experiment_config(data_dir: Path | str, name: str, config: dict, protocol_manifest: dict) -> dict:
    """Commit validation-selected hyperparameters before any fresh test scoring."""
    if not name or not all(character.isalnum() or character in "-_" for character in name):
        raise ValueError("Experiment name must contain only letters, digits, hyphens and underscores")
    payload = {"protocol_sha256": public_summary(protocol_manifest)["protocol_sha256"],
               "selection_source": "validation-only; frozen before fresh-user test",
               "config": config}
    return _write_once(Path(data_dir) / "protocols" / f"{name}.frozen-config.json", payload)
