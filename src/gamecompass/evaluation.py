"""Seeded evaluation with entire held-out users and full eligible-catalog ranking."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import copy
import json
import logging
from pathlib import Path
import sqlite3
import time
from typing import Mapping

import numpy as np
from scipy import sparse

from .models import ALGORITHMS, Model, atomic_json, sha256

LOGGER = logging.getLogger(__name__)

HISTORY_LENGTHS = (0, 1, 3, 5, 10, 20, 50, 100)
Row = tuple[str, str, int, int]


def user_partition(user_id: str, seed: int = 42) -> str:
    bucket = int(hashlib.sha256(f"{seed}:{user_id}".encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def prepare(data_dir: Path, max_items: int = 1500, seed: int = 42) -> dict:
    """Use training users alone to select item scope; no held-out-user filtering leak."""
    db = sqlite3.connect(data_dir / "population.sqlite")
    db.row_factory = sqlite3.Row
    games = [dict(row) for row in db.execute("SELECT * FROM games ORDER BY game_id")]
    available = {g["game_id"] for g in games}
    groups: dict[str, list[Row]] = defaultdict(list)
    for row in db.execute("SELECT user_id,game_id,positive,COALESCE(timestamp,0) FROM interactions ORDER BY user_id,COALESCE(timestamp,0),game_id"):
        if row[1] in available:
            groups[row[0]].append(tuple(row))
    db.close()
    partitions = {name: {} for name in ("train", "validation", "test")}
    for user, rows in groups.items():
        partitions[user_partition(user, seed)][user] = rows
    # Warm holdout is removed before item selection, statistics or fitting.
    warm = {}
    train_rows = []
    for user, rows in partitions["train"].items():
        positives = [i for i, row in enumerate(rows) if row[2] == 1]
        if len(positives) >= 3:
            idx = positives[-1]
            warm[user] = [rows[idx]]
            train_rows.extend(row for i, row in enumerate(rows) if i != idx)
        else:
            train_rows.extend(rows)
    counts = Counter(row[1] for row in train_rows if row[2] == 1)
    cf_ids = [game for game, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:max_items]]
    if len(cf_ids) < 2:
        raise ValueError("Need at least two games with positive training interactions")
    eligible = set(cf_ids)
    train_rows = [row for row in train_rows if row[1] in eligible]
    for split in ("validation", "test"):
        partitions[split] = {u: [r for r in rows if r[1] in eligible] for u, rows in partitions[split].items()}
        partitions[split] = {u: r for u, r in partitions[split].items() if r}
    warm = {u: r for u, r in warm.items() if r[0][1] in eligible}
    return {"games": games, "cf_ids": cf_ids, "train_rows": train_rows,
            "validation_users": partitions["validation"], "test_users": partitions["test"],
            "warm_holdouts": warm, "seed": seed, "max_items": max_items,
            "source_users": len(groups), "source_interactions": sum(map(len, groups.values()))}


def interaction_matrix(rows: list[Row], ids: list[str]) -> sparse.csr_matrix:
    items = {g: i for i, g in enumerate(ids)}
    users = {u: i for i, u in enumerate(sorted({r[0] for r in rows}))}
    positions = [(users[u], items[g]) for u, g, positive, _ in rows if positive and g in items]
    if not positions:
        raise ValueError("No positive training interactions")
    rr, cc = zip(*positions)
    return sparse.csr_matrix((np.ones(len(rr), dtype=np.float32), (rr, cc)), shape=(len(users), len(ids)))


def ranked_indices(model: Model, support: Mapping[str, float], seen: set[str], algorithm: str, k: int = 20) -> np.ndarray:
    scores = model.score(support, algorithm)
    candidates = np.array([model._idx[g] for g in model.cf_game_ids if g not in seen], dtype=int)
    # Stable ties resolve by canonical ID, identically for every algorithm.
    return candidates[np.lexsort((np.array(model.game_ids)[candidates], -scores[candidates]))[:k]]


def ranking_metrics(ranking: list[str], relevant: set[str], k: int) -> dict:
    if not relevant:
        raise ValueError("A ranking query needs at least one positive target")
    hits = np.array([g in relevant for g in ranking[:k]], dtype=float)
    n = len(hits)
    denom = np.log2(np.arange(n) + 2)
    ideal = np.sum(1 / np.log2(np.arange(min(k, len(relevant))) + 2))
    ap = np.sum(np.cumsum(hits) / (np.arange(n) + 1) * hits) / min(k, len(relevant))
    return {f"recall@{k}": float(hits.sum() / len(relevant)),
            f"ndcg@{k}": float((hits / denom).sum() / ideal), f"map@{k}": float(ap)}


def evaluate_queries(model: Model, queries: list[tuple[dict, set, set, set]], algorithm: str, seed: int = 42) -> dict:
    """queries: signed support, seen IDs, positive targets, explicit negative targets."""
    records, recommendations = [], set()
    pop = model.popularity[model._cf_out]
    total = float(pop.sum())
    cutoff = float(np.quantile(pop, 0.8))
    tail_ids = {g for g, count in zip(model.cf_game_ids, pop) if count <= cutoff}
    for support, seen, target, negatives in queries:
        ranked = ranked_indices(model, support, seen, algorithm)
        ranking = [model.game_ids[i] for i in ranked]
        record = {**ranking_metrics(ranking, target, 10), **ranking_metrics(ranking, target, 20)}
        record["explicit_negative_fraction@20"] = len(set(ranking) & negatives) / max(1, len(ranking))
        tail_targets = target & tail_ids
        record["long_tail_recall@20"] = len(set(ranking) & tail_targets) / len(tail_targets) if tail_targets else None
        counts = model.popularity[ranked]
        record["novelty@20_bits"] = float(np.mean(-np.log2((counts + 1) / (total + len(model.cf_game_ids))))) if len(ranked) else 0.0
        if len(ranked) > 1:
            features = model.content[ranked]
            features = features[np.asarray(features.getnnz(axis=1)).ravel() > 0]
            similarities = (features @ features.T).toarray()
            count = features.shape[0]
            record["diversity@20"] = float(1 - (similarities.sum() - np.trace(similarities)) / (count * (count - 1))) if count > 1 else None
        else:
            record["diversity@20"] = 0.0
        records.append(record)
        recommendations.update(ranking)
    if not records:
        return {"users": 0, "status": "insufficient history in this cohort"}
    metrics = {}
    rng = np.random.default_rng(seed)
    for key in records[0]:
        values = np.array([r[key] for r in records if r[key] is not None])
        if not len(values):
            metrics[key] = None
            continue
        bootstrap = np.mean(values[rng.integers(0, len(values), size=(200, len(values)))], axis=1)
        metrics[key] = {"mean": float(values.mean()), "ci95": [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])], "n": len(values)}
    return {"users": len(records), "metrics": metrics, "catalog_coverage@20": len(recommendations) / len(model.cf_game_ids)}


def cold_queries(users: dict[str, list[Row]], history: int, max_users: int = 500) -> list:
    queries = []
    for user in sorted(users):
        rows = users[user]
        if len(rows) <= history:
            continue
        support_rows, heldout = rows[:history], rows[history:]
        target = {r[1] for r in heldout if r[2]}
        if not target:
            continue
        support = {g: 1.0 if positive else -1.0 for _, g, positive, _ in support_rows}
        queries.append((support, {r[1] for r in support_rows}, target, {r[1] for r in heldout if not r[2]}))
        if len(queries) >= max_users:
            break
    return queries


def run_benchmark(data_dir: Path, max_items: int = 1500, seed: int = 42, max_users: int = 500) -> dict:
    started = time.monotonic()
    data_dir = Path(data_dir)
    LOGGER.info("Preparing deterministic train/validation/test users")
    prepared = prepare(data_dir, max_items, seed)
    LOGGER.info("Fitting population models on %d interactions and %d items", len(prepared["train_rows"]), len(prepared["cf_ids"]))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    version = f"development-{max_items}-{seed}-{timestamp}"
    experiment = data_dir / "experiments" / version
    experiment.mkdir(parents=True)
    atomic_json(experiment / "split.json", {k: v for k, v in prepared.items() if k != "games"})
    model = Model.fit(prepared["games"], prepared["cf_ids"], interaction_matrix(prepared["train_rows"], prepared["cf_ids"]), seed)
    model.metadata.update({"scope": "development bounded training-user-selected item universe", "max_items": max_items,
                           "source_sha256": sha256(data_dir / "population.sqlite"), "split_sha256": sha256(experiment / "split.json"),
                           "version": version, "validation_user_count": len(prepared["validation_users"]),
                           "test_user_count": len(prepared["test_users"])})
    full_model = model
    model = copy.copy(full_model)
    model.game_ids = list(full_model.cf_game_ids)
    model.popularity = full_model.popularity[full_model._cf_out]
    model.content = full_model.content[full_model._cf_out]
    model._indexes()
    LOGGER.info("Evaluating validation users and selecting mixture weights")
    # Tune only held-out validation users with 1/3/5 support interactions.
    val_queries = [q for history in (1, 3, 5) for q in cold_queries(prepared["validation_users"], history, max_users)]
    if not val_queries:
        raise ValueError("Insufficient held-out validation users; cannot select production model")
    validation = {}
    for algorithm in ALGORITHMS[:-1]:
        validation[algorithm] = evaluate_queries(model, val_queries, algorithm, seed)
    cf_winner = max(("itemknn", "ease", "svd"), key=lambda a: validation[a]["metrics"]["ndcg@20"]["mean"])
    model.hybrid_cf = cf_winner
    # Cache validation component scores. Weight selection never reads test targets.
    components = []
    for support, seen, target, _ in val_queries:
        valid = np.array([model._idx[g] for g in model.cf_game_ids if g not in seen])
        components.append((model._scaled(model.score(support, cf_winner)),
                           model._scaled(model.score(support, "content")), valid, target))
    hybrid_trials = []
    prior = model._scaled(model.popularity)
    for alpha in (0.0, 0.05, 0.1, 0.25, 0.5):
        for popularity_weight in (0.0, 0.1, 0.25, 0.5, 0.75):
            if alpha + popularity_weight >= 1:
                continue
            values = []
            for cf, content, valid, target in components:
                score = (1 - alpha - popularity_weight) * cf + alpha * content + popularity_weight * prior
                ranked = valid[np.lexsort((np.array(model.game_ids)[valid], -score[valid]))[:20]]
                values.append(ranking_metrics([model.game_ids[i] for i in ranked], target, 20)["ndcg@20"])
            hybrid_trials.append({"alpha": alpha, "popularity_weight": popularity_weight,
                                  "validation_ndcg@20": float(np.mean(values))})
    best_hybrid = max(hybrid_trials, key=lambda t: t["validation_ndcg@20"])
    model.hybrid_alpha = best_hybrid["alpha"]
    model.hybrid_popularity = best_hybrid["popularity_weight"]
    validation["hybrid"] = evaluate_queries(model, val_queries, "hybrid", seed)
    # Deterministic tie-break prefers simpler earlier algorithms.
    model.selected = max(ALGORITHMS, key=lambda a: validation[a]["metrics"]["ndcg@20"]["mean"])
    history_validation = {}
    for history in HISTORY_LENGTHS:
        queries = cold_queries(prepared["validation_users"], history, max_users)
        if len(queries) < 30:
            history_validation[str(history)] = {"queries": len(queries), "status": "fewer than 30 validation queries; use pooled winner"}
            continue
        results = {a: evaluate_queries(model, queries, a, seed) for a in ALGORITHMS}
        winner = max(ALGORITHMS, key=lambda a: results[a]["metrics"]["ndcg@20"]["mean"])
        model.selection_by_history[str(history)] = winner
        history_validation[str(history)] = {"queries": len(queries), "selected": winner, "results": results}
    cold = {}
    for history in HISTORY_LENGTHS:
        LOGGER.info("Evaluating cold-test support length %d", history)
        queries = cold_queries(prepared["test_users"], history, max_users)
        cold[str(history)] = {algorithm: evaluate_queries(model, queries, algorithm, seed) for algorithm in ALGORITHMS}
    production_cold = {}
    for history, results in cold.items():
        eligible_bins = [int(h) for h in model.selection_by_history if int(h) <= int(history)]
        algorithm = model.selection_by_history[str(max(eligible_bins))] if eligible_bins else model.selected
        production_cold[history] = {"algorithm": algorithm, "result": results[algorithm]}
    train_by_user: dict[str, list[Row]] = defaultdict(list)
    for row in prepared["train_rows"]:
        train_by_user[row[0]].append(row)
    warm_queries = []
    for user in sorted(prepared["warm_holdouts"]):
        rows = train_by_user[user]
        if not rows:
            continue
        support = {g: 1.0 if positive else -1.0 for _, g, positive, _ in rows}
        warm_queries.append((support, set(support), {r[1] for r in prepared["warm_holdouts"][user]}, set()))
        if len(warm_queries) >= max_users:
            break
    warm = {algorithm: evaluate_queries(model, warm_queries, algorithm, seed) for algorithm in ALGORITHMS}
    artifact_dir = data_dir / "models" / version
    full_model.selected = model.selected
    full_model.hybrid_alpha = model.hybrid_alpha
    full_model.hybrid_cf = model.hybrid_cf
    full_model.hybrid_popularity = model.hybrid_popularity
    full_model.selection_by_history = model.selection_by_history
    full_model.save(artifact_dir)
    atomic_json(artifact_dir / "evaluation.json", {"validation_ndcg_at_20": validation[model.selected]["metrics"]["ndcg@20"]["mean"], "selection_source": "held-out validation users only", "split_sha256": model.metadata["split_sha256"]})
    stats_path = data_dir / "dataset-stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    active_path = data_dir / "active-population.json"
    if active_path.exists():
        active = json.loads(active_path.read_text())
        if active.get("sha256") != model.metadata["source_sha256"]:
            raise ValueError("Active population provenance does not match trained database")
        source = active.get("source", {})
        for candidate in (data_dir / "kaggle-population.stats.json", stats_path):
            if candidate.exists():
                candidate_stats = json.loads(candidate.read_text())
                if candidate_stats.get("source", {}).get("ref") == source.get("ref"):
                    stats = candidate_stats
                    break
        stats["source"] = source
    report = {"seed": seed, "configured_max_items": max_items, "dataset": stats.get("source", {}), "data_cohort_selection": stats.get("selection", "see dataset provenance"), "version": version, "dataset_sha256": model.metadata["source_sha256"], "split_sha256": model.metadata["split_sha256"],
              "scope": model.metadata["scope"], "catalog_items": len(full_model.game_ids), "eligible_cf_items": len(model.cf_game_ids),
              "training_users": model.metadata["training_users"], "training_positives": model.metadata["training_positives"],
              "source_users": prepared["source_users"], "source_interactions": prepared["source_interactions"],
              "max_evaluation_users_per_cohort": max_users,
              "selection_objective": "pooled validation held-out users support lengths 1,3,5 NDCG@20; same user may have multiple support queries",
              "selected": model.selected, "validation": validation, "hybrid_trials": hybrid_trials,
              "selection_by_history": model.selection_by_history, "history_validation": history_validation,
              "cold_test": cold, "production_cold_test": production_cold, "warm_test": warm,
              "temporal_evaluation": "not run; support histories chronological but population model may contain later events from other users",
              "candidate_strategy": "full training-user-selected eligible item universe, no sampled negatives; all support games excluded; other unobserved items are ranking distractors, not explicit dislikes",
              "long_tail_definition": "training-positive count at/below 80th percentile among eligible items",
              "ci_method": "200 user/query bootstrap samples, percentile 95%; pooled validation repeated users not independent",
              "duration_seconds": time.monotonic() - started,
              "limitations": ["Historical Steam review-selection bias; not a probability of real-world enjoyment", "Item-universe restriction omits cold and low-count items", "Metadata snapshot is static and may postdate historical interactions", "Warm holdout tests item fold-in; no per-user stored latent vectors", "Single seeded split; no significance claim from overlapping confidence intervals", "No global chronological temporal benchmark yet", "No semantic embedding comparison yet"]}
    atomic_json(experiment / "results.json", report)
    atomic_json(data_dir / "models" / "current.json", {"version": version})
    atomic_json(data_dir / "benchmark.json", report)
    write_report(report, Path(__file__).resolve().parents[2] / "docs" / "benchmarks.md")
    return report


def write_report(report: dict, path: Path) -> None:
    lines = ["# Measured development benchmark", "", f"Artifact: `{report['version']}`.", "", "This is a real-data development experiment, not a full-dataset accuracy claim.", "", f"Dataset: {report.get('dataset', {}).get('ref', 'see provenance report')}; version {report.get('dataset', {}).get('version', 'see provenance')}. Cohort: {report.get('data_cohort_selection', 'see provenance')}.", "",
             f"Population: {report['source_users']:,} users / {report['source_interactions']:,} normalized reviews; {report['training_users']:,} training users / {report['training_positives']:,} positive training reviews.",
             f"Local catalog: {report['catalog_items']:,}; eligible ranking universe: {report['eligible_cf_items']:,} games selected only from training-positive counts.",
             f"Elapsed: {report['duration_seconds']:.1f} seconds. Selected by validation: **{report['selected']}**.", "",
             "## Protocol", "", report["candidate_strategy"], "", report["selection_objective"], "", "User assignment: seeded SHA-256 buckets, 80% population training / 10% validation / 10% test. Final positive of training users with >=3 positives removed for a separate warm fold-in test before item selection. Cold-test users never contribute training interactions, item popularity, CF weights or latent factors. Support is the chronological prefix of explicit reviews, including signed negatives; target is remaining positive reviews. History lengths count reviews only within the eligible item universe; interactions outside that scope are not evaluated. Rows with unknown timestamps sort first. Cohorts require an unseen positive target. All algorithms share each cohort and candidate list. Cohorts may contain different users. Maximum first sorted users per cohort is configured; it is not random sampling.", "",
             "Recall/NDCG/MAP at 10 and 20, explicit-negative fraction, novelty, metadata diversity and long-tail recall plus bootstrap intervals are in `data/benchmark.json`. Confidence intervals are descriptive, not evidence that any model wins statistically.", "", "## Cold test NDCG@20 (Recall@20)", "", "| Support | Queries | " + " | ".join(ALGORITHMS) + " |", "|---|---|" + "---|" * len(ALGORITHMS)]
    for history, results in report["cold_test"].items():
        cells = []
        for algorithm in ALGORITHMS:
            result = results[algorithm]
            cells.append("n/a" if not result["users"] else f"{result['metrics']['ndcg@20']['mean']:.4f} ({result['metrics']['recall@20']['mean']:.4f})")
        lines.append(f"| {history} | {results['popularity']['users']} | " + " | ".join(cells) + " |")
    lines += ["", "## Validation", "", "| Algorithm | NDCG@20 | Recall@20 |", "|---|---:|---:|"]
    for algorithm, result in report["validation"].items():
        lines.append(f"| {algorithm} | {result['metrics']['ndcg@20']['mean']:.4f} | {result['metrics']['recall@20']['mean']:.4f} |")
    lines += ["", "## Validation-selected routing", "", "The pooled winner is used when no eligible history bucket exists. Otherwise the algorithm selected on at least 30 validation queries for the nearest lower history bucket is used; test results do not choose routes.", "", "| Minimum explicit history | Algorithm |", "|---|---|"]
    lines += [f"| {n} | {a} |" for n, a in report.get("selection_by_history", {}).items()]
    lines += ["", "## Production routing on untouched test users", "", "| Support | Algorithm | Queries | NDCG@20 | Recall@20 |", "|---|---|---:|---:|---:|"]
    for n, routed in report.get("production_cold_test", {}).items():
        result = routed["result"]
        if result["users"]:
            lines.append(f"| {n} | {routed['algorithm']} | {result['users']} | {result['metrics']['ndcg@20']['mean']:.4f} | {result['metrics']['recall@20']['mean']:.4f} |")
        else:
            lines.append(f"| {n} | {routed['algorithm']} | 0 | n/a | n/a |")
    lines += ["", "## Limits", ""] + [f"- {item}" for item in report["limitations"]]
    lines += ["", "No numerical-rating RMSE/MAE: source feedback is binary recommendation, not a valid star-rating scale.", "", "Source code licensing is separate from data and trained weights. Consult the dataset provenance and licensing report before distributing artifacts.", "", f"Dataset SHA-256: `{report['dataset_sha256']}`", f"Split SHA-256: `{report['split_sha256']}`", "", f"Reproduce: `gamecompass benchmark --max-items {report['eligible_cf_items']} --max-users {report['max_evaluation_users_per_cohort']} --seed {report.get('seed', 42)}`."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    atomic_json(path.parent / "benchmark-results.json", report)


def run_temporal_benchmark(data_dir: Path, max_items: int = 1500, seed: int = 42, max_users: int = 300) -> dict:
    """Global chronological cutoff: every CF training event predates every target.

    Catalog text is a present-day snapshot; content results are explicitly not
    evidence of historically available metadata. No parameters selected here.
    """
    started = time.monotonic()
    db = sqlite3.connect(Path(data_dir) / "population.sqlite")
    db.row_factory = sqlite3.Row
    games = [dict(row) for row in db.execute("SELECT * FROM games ORDER BY game_id")]
    games_by_id = {g["game_id"]: g for g in games}
    rows = [tuple(row) for row in db.execute("SELECT user_id,game_id,positive,timestamp FROM interactions WHERE timestamp > 0 ORDER BY user_id,timestamp,game_id") if row[1] in games_by_id]
    db.close()
    if len(rows) < 10:
        return {"status": "insufficient valid timestamps", "timestamped_interactions": len(rows)}
    cutoff = int(np.quantile([r[3] for r in rows], .8))
    prior = [r for r in rows if r[3] <= cutoff]
    future = [r for r in rows if r[3] > cutoff]
    counts = Counter(r[1] for r in prior if r[2])
    ids = [g for g, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:max_items]]
    eligible = set(ids)
    prior = [r for r in prior if r[1] in eligible]
    model = Model.fit([games_by_id[g] for g in ids], ids, interaction_matrix(prior, ids), seed)
    # Use previously validated hybrid settings; no temporal test-set tuning.
    pointer = Path(data_dir) / "models" / "current.json"
    if pointer.exists():
        old_version = json.loads(pointer.read_text())["version"]
        old_manifest = json.loads((Path(data_dir) / "models" / old_version / "manifest.json").read_text())
        model.hybrid_alpha = old_manifest["hybrid_alpha"]
        model.hybrid_cf = old_manifest["hybrid_cf"]
        model.hybrid_popularity = old_manifest.get("hybrid_popularity", 0.0)
    users_prior: dict[str, list[Row]] = defaultdict(list)
    users_future: dict[str, list[Row]] = defaultdict(list)
    for row in prior:
        users_prior[row[0]].append(row)
    for row in future:
        if row[1] in eligible:
            users_future[row[0]].append(row)
    queries = []
    for user in sorted(users_future):
        support = {g: 1.0 if positive else -1.0 for _, g, positive, _ in users_prior[user]}
        target = {r[1] for r in users_future[user] if r[2] and r[1] not in support}
        if not target:
            continue
        negatives = {r[1] for r in users_future[user] if not r[2]}
        queries.append((support, set(support), target, negatives))
        if len(queries) >= max_users:
            break
    results = {algorithm: evaluate_queries(model, queries, algorithm, seed) for algorithm in ALGORITHMS}
    report = {"dataset_sha256": sha256(Path(data_dir) / "population.sqlite"), "seed": seed, "max_items": max_items, "max_users": max_users, "cutoff_timestamp": cutoff, "cutoff_utc": datetime.fromtimestamp(cutoff, timezone.utc).isoformat(),
              "timestamped_interactions": len(rows), "training_interactions": len(prior),
              "future_interactions": len(future), "eligible_items": len(ids), "evaluation_queries": len(queries),
              "future_item_scope_coverage": sum(r[1] in eligible for r in future) / max(1, len(future)),
              "results": results, "duration_seconds": time.monotonic() - started,
              "protocol": "global 80th-percentile timestamp cutoff, all CF/popularity fitting at/before cutoff; targets strictly later; no tuning on temporal targets; full eligible item ranking excluding prior reviews",
              "limitations": ["Static metadata may postdate cutoff; temporal content/hybrid results have metadata look-ahead limitation", "Eligible items selected only from prior positives; future unseen items excluded and coverage reported", "First sorted eligible users capped by max_users, not random sampling"]}
    atomic_json(Path(data_dir) / "temporal-benchmark.json", report)
    docs = Path(__file__).resolve().parents[2] / "docs"
    atomic_json(docs / "temporal-benchmark-results.json", report)
    lines = ["# Global chronological development benchmark", "", report["protocol"], "",
             f"Cutoff: {report['cutoff_utc']}. Timestamped rows: {len(rows):,}; fitting interactions: {len(prior):,}; later interactions: {len(future):,}.",
             f"Eligible games: {len(ids):,}; evaluated user queries: {len(queries)}; future interaction item-scope coverage: {report['future_item_scope_coverage']:.2%}.", "",
             "| Algorithm | Recall@20 | NDCG@20 | NDCG 95% CI |", "|---|---:|---:|---|"]
    for algorithm, result in results.items():
        if not result['users']:
            continue
        metric = result['metrics']['ndcg@20']
        lines.append(f"| {algorithm} | {result['metrics']['recall@20']['mean']:.4f} | {metric['mean']:.4f} | {metric['ci95'][0]:.4f}–{metric['ci95'][1]:.4f} |")
    lines += ["", "This is a separate temporal test; parameters are fixed or inherited from validation. It is not used for model selection. Existing users have only prior reviews as support; future-only users receive the population baseline. All methods use identical queries and candidate lists.", ""]
    lines += [f"- {limit}" for limit in report['limitations']]
    lines += ["", f"Dataset SHA-256: `{report['dataset_sha256']}`.", "", "Full aggregate results: temporal-benchmark-results.json. Reproduce: `gamecompass benchmark-temporal --max-items 1500 --max-users 300 --seed 42`."]
    (docs / "benchmarks-temporal.md").write_text("\n".join(lines) + "\n")
    return report


def implicit_queries(users: dict, history: int, weighting: str, max_users: int = 300) -> list:
    """Library proxy: reveal reviewed game IDs/hours while hiding support labels."""
    if weighting not in {"uniform", "log1p_hours", "explicit"}:
        raise ValueError("Unknown implicit weighting")
    queries = []
    for user in sorted(users):
        rows = users[user]
        if len(rows) <= history:
            continue
        known, heldout = rows[:history], rows[history:]
        targets = {r[1] for r in heldout if r[2]}
        if not targets:
            continue
        support = {}
        for _, game, positive, _, hours in known:
            if weighting == "uniform":
                weight = 1.0
            elif weighting == "log1p_hours":
                weight = float(np.log1p(max(0, float(hours or 0))))
            else:
                weight = 1.0 if positive else -1.0
            support[game] = weight
        queries.append((support, set(support), targets, {r[1] for r in heldout if not r[2]}))
        if len(queries) >= max_users:
            break
    return queries


def run_implicit_benchmark(data_dir: Path, seed: int = 42, max_users: int = 300) -> dict:
    """Select an implicit-only adapter using validation users; do not retrain CF.

    This is a proxy for library imports, not a representative ownership dataset:
    all source support games were reviewed, and source hours may have accrued
    after review timestamps. The limitation is deliberately retained in results.
    """
    data_dir = Path(data_dir)
    version = json.loads((data_dir / "models" / "current.json").read_text())["version"]
    artifact = data_dir / "models" / version
    full_model = Model.load(artifact)
    if full_model.metadata["source_sha256"] != sha256(data_dir / "population.sqlite"):
        raise ValueError("Implicit adapter population differs from the active model")
    model = copy.copy(full_model)
    model.game_ids = list(full_model.cf_game_ids)
    model.content = full_model.content[full_model._cf_out]
    model.popularity = full_model.popularity[full_model._cf_out]
    model._indexes()
    users = {"validation": defaultdict(list), "test": defaultdict(list)}
    eligible = set(model.cf_game_ids)
    db = sqlite3.connect(data_dir / "population.sqlite")
    partitions = {}
    for row in db.execute("SELECT user_id,game_id,positive,COALESCE(timestamp,0),playtime FROM interactions ORDER BY user_id,COALESCE(timestamp,0),game_id"):
        user, game, _, _, _ = row
        if game not in eligible:
            continue
        if user not in partitions:
            partitions[user] = user_partition(user, seed)
        split = partitions[user]
        if split != "train":
            users[split][user].append(tuple(row))
    db.close()
    algorithms = ("popularity", "itemknn", "ease", "svd", "content", "hybrid")
    weightings = ("uniform", "log1p_hours")
    validation = {}
    pooled = Counter()
    pooled_count = Counter()
    choices = {}
    for history in HISTORY_LENGTHS:
        validation[str(history)] = {}
        for weighting in weightings:
            queries = implicit_queries(users["validation"], history, weighting, max_users)
            if not queries:
                continue
            for algorithm in algorithms:
                key = f"{weighting}:{algorithm}"
                result = evaluate_queries(model, queries, algorithm, seed)
                value = result["metrics"]["ndcg@20"]["mean"]
                validation[str(history)][key] = {"queries": len(queries), "ndcg@20": value}
                if history in (1, 3, 5):
                    pooled[key] += value * len(queries)
                    pooled_count[key] += len(queries)
        if validation[str(history)] and next(iter(validation[str(history)].values()))["queries"] >= 30:
            choices[str(history)] = max(validation[str(history)], key=lambda k: validation[str(history)][k]["ndcg@20"])
    if not pooled:
        raise ValueError("Insufficient validation support for implicit adapter")
    best = max(pooled, key=lambda k: pooled[k] / pooled_count[k])
    selected = {"weighting": best.split(":")[0], "algorithm": best.split(":")[1],
                "selection_by_history": {h: {"weighting": key.split(":")[0], "algorithm": key.split(":")[1]} for h, key in choices.items()},
                "source": "validation-only reviewed-game library proxy; labels hidden in support; hours from source snapshot",
                "seed": seed, "mixed_explicit_implicit": "not validated; explicit enjoyment feedback takes precedence"}
    test = {}
    for history in HISTORY_LENGTHS:
        eligible_bins = [int(h) for h in choices if int(h) <= history]
        choice = selected["selection_by_history"][str(max(eligible_bins))] if eligible_bins else selected
        queries = implicit_queries(users["test"], history, choice["weighting"], max_users)
        explicit = implicit_queries(users["test"], history, "explicit", max_users)
        explicit_bins = [int(h) for h in full_model.selection_by_history if int(h) <= history]
        explicit_algorithm = full_model.selection_by_history[str(max(explicit_bins))] if explicit_bins else full_model.selected
        test[str(history)] = {"selected_weighting": choice["weighting"], "selected_algorithm": choice["algorithm"],
                              "selected": evaluate_queries(model, queries, choice["algorithm"], seed),
                              "popularity": evaluate_queries(model, queries, "popularity", seed),
                              "explicit_reference": evaluate_queries(model, explicit, explicit_algorithm, seed)}
    report = {"model_version": version, "population_sha256": full_model.metadata["source_sha256"],
              "max_users": max_users, "selected": selected, "validation": validation, "cold_test": test,
              "protocol": "entire validation/test users held out from CF training; reviewed game IDs and source hours revealed without support recommendation labels; remaining positive reviews as targets; full eligible item ranking; support excluded",
              "limitations": ["Reviewed games are an exposure-selected proxy for a game library; ownership-only real-world quality is unverified", "Snapshot playtime may include activity after historical review date; this is not a temporal implicit benchmark", "Playtime and ownership are weaker signals than explicit enjoyment; no label conversion in persistent data", "Mixed explicit-plus-implicit weighting has not been validated"]}
    # Add only adapter configuration to the immutable-weight model manifest.
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["implicit"] = selected
    atomic_json(manifest_path, manifest)
    atomic_json(data_dir / "implicit-benchmark.json", report)
    docs = Path(__file__).resolve().parents[2] / "docs"
    atomic_json(docs / "implicit-benchmark-results.json", report)
    lines = ["# Implicit-only personalization adapter", "", report["protocol"], "", "This experiment hides support labels and is different from explicit enjoyment personalization. It does not equate ownership or hours with liking a game.", "",
             "Validation compares uniform observed-game weights with log1p(hours), across popularity, item-kNN, EASE, SVD, content and the validated hybrid. Select a weighting and algorithm per history bucket only when at least 30 validation queries exist; otherwise use the closest validated lower bucket or pooled winner.", "",
             "| History | Test queries | Adapter | NDCG@20 | Recall@20 | Popularity NDCG@20 | Explicit reference NDCG@20 |", "|---|---:|---|---:|---:|---:|---:|"]
    for h, result in test.items():
        picked = result["selected"]
        if not picked["users"]:
            lines.append(f"| {h} | 0 | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(f"| {h} | {picked['users']} | {result['selected_weighting']} + {result['selected_algorithm']} | {picked['metrics']['ndcg@20']['mean']:.4f} | {picked['metrics']['recall@20']['mean']:.4f} | {result['popularity']['metrics']['ndcg@20']['mean']:.4f} | {result['explicit_reference']['metrics']['ndcg@20']['mean']:.4f} |")
    lines += ["", "## Limits", ""] + [f"- {item}" for item in report["limitations"]]
    lines += ["", "Full validation/test aggregates and confidence intervals: implicit-benchmark-results.json. No raw histories are included in this report."]
    (docs / "benchmarks-implicit.md").write_text("\n".join(lines) + "\n")
    return report


def cold_item_partition(game_id: str, seed: int = 42) -> bool:
    """True means interaction evidence is withheld for this item in the experiment."""
    return int(hashlib.sha256(f"cold-item:{seed}:{game_id}".encode()).hexdigest()[:8], 16) % 5 == 0


def cold_item_queries(users: dict[str, list[Row]], cold_ids: set[str], history: int, max_users: int = 300) -> list:
    if history < 1:
        raise ValueError("Cold-item prefix experiment requires positive support length")
    queries = []
    for user in sorted(users):
        rows = users[user]
        warm_positions = [i for i, row in enumerate(rows) if row[1] not in cold_ids]
        if len(warm_positions) < history:
            continue
        positions = warm_positions[:history]
        support = {rows[i][1]: 1.0 if rows[i][2] else -1.0 for i in positions}
        later = rows[positions[-1] + 1:]
        targets = {r[1] for r in later if r[1] in cold_ids and r[2]}
        if not targets:
            continue
        queries.append((support, set(support), targets, {r[1] for r in later if r[1] in cold_ids and not r[2]}))
        if len(queries) >= max_users:
            break
    return queries


def expected_random_metrics(target_count: int, candidate_count: int, k: int) -> dict:
    """Analytic expectation under uniformly random item ordering, not a measured run."""
    limit = min(k, candidate_count)
    discounts = 1 / np.log2(np.arange(limit) + 2)
    ideal = np.sum(1 / np.log2(np.arange(min(k, target_count)) + 2))
    precision_hits = sum((target_count / candidate_count) * (1 + (i - 1) * (target_count - 1) / max(1, candidate_count - 1)) / i for i in range(1, limit + 1))
    return {f"recall@{k}": limit / candidate_count,
            f"ndcg@{k}": float(target_count / candidate_count * discounts.sum() / ideal),
            f"map@{k}": precision_hits / min(k, target_count)}


def run_cold_item_benchmark(data_dir: Path, seed: int = 42, max_users: int = 300) -> dict:
    """Metadata-only cold-item ranking with every cold collaborative signal unused.

    We intentionally never read the fitted CF scores or popularity values here.
    Only metadata TF-IDF, externally fixed item hashes and test-user support are
    used. A zero-evidence CF model and a uniform prior are identical baselines.
    """
    data_dir = Path(data_dir)
    version = json.loads((data_dir / "models" / "current.json").read_text())["version"]
    model = Model.load(data_dir / "models" / version)
    source_hash = sha256(data_dir / "population.sqlite")
    if source_hash != model.metadata["source_sha256"]:
        raise ValueError("Cold-item catalog source differs from the active model")
    cold_ids = {g for g in model.cf_game_ids if cold_item_partition(g, seed)}
    if len(cold_ids) < 20:
        raise ValueError("Need at least 20 item-disjoint cold candidates")
    candidate_ids = sorted(cold_ids)
    candidates = np.array([model._idx[g] for g in candidate_ids])
    # Read only held-out test users; no test labels set hyperparameters.
    users: dict[str, list[Row]] = defaultdict(list)
    eligible = set(model.cf_game_ids)
    db = sqlite3.connect(data_dir / "population.sqlite")
    partitions = {}
    for row in db.execute("SELECT user_id,game_id,positive,COALESCE(timestamp,0) FROM interactions ORDER BY user_id,COALESCE(timestamp,0),game_id"):
        if row[1] not in eligible:
            continue
        if row[0] not in partitions:
            partitions[row[0]] = user_partition(row[0], seed)
        if partitions[row[0]] == "test":
            users[row[0]].append(tuple(row))
    db.close()
    results = {}
    for history in (1, 3, 5):
        queries = cold_item_queries(users, cold_ids, history, max_users)
        records = {"content": [], "no_collaborative_evidence_uniform_prior": [], "analytic_expected_random_order": []}
        covered = {key: set() for key in records}
        for support, _, targets, negatives in queries:
            assert not set(support) & cold_ids
            support_indices = [model._idx[g] for g in support]
            centroid = np.asarray(model.content[support_indices].T @ np.array(list(support.values()), dtype=np.float32)).ravel()
            length = np.linalg.norm(centroid)
            if length:
                centroid /= length
            scores = np.asarray(model.content[candidates] @ centroid).ravel()
            content_rank = [candidate_ids[i] for i in np.lexsort((np.array(candidate_ids), -scores))[:20]]
            for key, ranking in (("content", content_rank), ("no_collaborative_evidence_uniform_prior", candidate_ids[:20])):
                record = {**ranking_metrics(ranking, targets, 10), **ranking_metrics(ranking, targets, 20),
                          "explicit_negative_fraction@20": len(set(ranking) & negatives) / 20}
                records[key].append(record)
                covered[key].update(ranking)
            records["analytic_expected_random_order"].append({**expected_random_metrics(len(targets), len(candidate_ids), 10),
                                                               **expected_random_metrics(len(targets), len(candidate_ids), 20)})
        results[str(history)] = {}
        for method, rows in records.items():
            if not rows:
                results[str(history)][method] = {"users": 0, "status": "insufficient cold-positive target queries"}
                continue
            metrics = {}
            rng = np.random.default_rng(seed)
            for key in rows[0]:
                values = np.array([r[key] for r in rows])
                bootstrap = np.mean(values[rng.integers(0, len(values), size=(200, len(values)))], axis=1)
                metrics[key] = {"mean": float(values.mean()), "ci95": [float(v) for v in np.quantile(bootstrap, [.025, .975])], "n": len(values)}
            result = {"users": len(rows), "metrics": metrics}
            if method != "analytic_expected_random_order":
                result["catalog_coverage@20"] = len(covered[method]) / len(candidate_ids)
            results[str(history)][method] = result
    report = {"model_version": version, "dataset_sha256": source_hash, "seed": seed, "max_users": max_users,
              "eligible_item_universe": len(model.cf_game_ids), "cold_items": len(cold_ids),
              "warm_items": len(model.cf_game_ids) - len(cold_ids), "results": results,
              "protocol": "hash20% bounded-universe item holdout; cold items never occur in support and no collaborative weights, interaction counts or popularity values are used for cold scoring; warm-only signed chronological prefix; later cold positive reviews as targets; full cold-item candidate ranking; complete test users absent from population fit",
              "training_note": "No additional CF model is trained: every cold-item CF prediction is defined to have no evidence (zero). Metadata TF-IDF is unsupervised and independent of interaction labels. Existing fitted CF weights are loaded as part of the artifact but never accessed for scoring.",
              "baseline_note": "Zero-evidence CF and uniform prior give identical zero scores with canonical-ID tie breaking. Analytic expected random order is a mathematical expectation, not an observed experiment.",
              "limitations": ["Conditional on users having a later cold positive; does not estimate all-user recommendation satisfaction", "Cold IDs are selected from a bounded training-positive-selected1500-item universe; not representative of truly unreleased games", "Metadata snapshot may postdate historical review timestamps; this is an item-evidence ablation, not a chronological release simulation", "No collaborative model is refit on warm items; only the cold-item no-evidence behavior is evaluated", "Hash partition and metadata parameters are fixed before test scoring; no test-tuned combination weights"]}
    atomic_json(data_dir / "cold-item-benchmark.json", report)
    docs = Path(__file__).resolve().parents[2] / "docs"
    atomic_json(docs / "cold-item-benchmark-results.json", report)
    lines = ["# Cold-item evidence ablation", "", report["protocol"], "", report["training_note"], "", report["baseline_note"], "",
             f"Cold candidate items: {len(cold_ids)} of {len(model.cf_game_ids)} eligible items. Candidate lists contain every cold item, without sampled negatives. Support history lengths count warm eligible reviews only.", "",
             "| Warm support | Queries | Content NDCG@20 (95% CI) | Content Recall@20 | Zero-evidence / uniform NDCG@20 | Expected random NDCG@20 |", "|---|---:|---|---:|---:|---:|"]
    for history, methods in results.items():
        result = methods["content"]
        if not result["users"]:
            lines.append(f"| {history} | 0 | n/a | n/a | n/a | n/a |")
            continue
        metric = result["metrics"]["ndcg@20"]
        lines.append(f"| {history} | {result['users']} | {metric['mean']:.4f} ({metric['ci95'][0]:.4f}–{metric['ci95'][1]:.4f}) | {result['metrics']['recall@20']['mean']:.4f} | {methods['no_collaborative_evidence_uniform_prior']['metrics']['ndcg@20']['mean']:.4f} | {methods['analytic_expected_random_order']['metrics']['ndcg@20']['mean']:.4f} |")
    lines += ["", "## Limits", ""] + [f"- {item}" for item in report["limitations"]]
    lines += ["", "This does not validate quality on actual new releases. Full metrics, confidence intervals and coverage: cold-item-benchmark-results.json.", "", "Reproduce: `python scripts/benchmark_cold_items.py --data-dir data --max-users 300 --seed 42`."]
    (docs / "benchmarks-cold-items.md").write_text("\n".join(lines) + "\n")
    return report
