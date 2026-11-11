"""Training-only aggregate review approval, distinct from personalized enjoyment.

This prior summarizes the opinions of observed reviewers, conditioned on source
selection/exposure. It is not a calibrated probability that a new user will like
an unplayed title. Neither review count nor approval alone represents taste.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .models import atomic_json, sha256


class QualityPrior:
    """Beta-shrunk observed approval and independent review-count channels.

    `fit` must receive population training rows *after* warm holdouts have been
    removed. Caller owns leakage-safe splitting. No user identifiers are saved.
    """

    def __init__(self) -> None:
        self.game_ids: list[str] = []
        self.positive = np.empty(0, dtype=np.int64)
        self.negative = np.empty(0, dtype=np.int64)
        self.global_rate = 0.5
        self.metadata: dict = {}
        self._index: dict[str, int] = {}

    @classmethod
    def fit(cls, rows: Iterable[Sequence], ids: Sequence[str]) -> "QualityPrior":
        prior = cls()
        prior.game_ids = list(ids)
        if any(not isinstance(gid, str) or not gid for gid in prior.game_ids):
            raise ValueError("Quality IDs must be nonempty canonical strings")
        if len(set(prior.game_ids)) != len(prior.game_ids):
            raise ValueError("Duplicate quality game IDs")
        prior._index = {gid: i for i, gid in enumerate(prior.game_ids)}
        prior.positive = np.zeros(len(ids), dtype=np.int64)
        prior.negative = np.zeros(len(ids), dtype=np.int64)
        outside = 0
        for row in rows:
            if len(row) < 3:
                raise ValueError("Quality rows require user, game, explicit binary label")
            gid, label = row[1], row[2]
            if label not in (0, 1):
                raise ValueError("Only explicit binary recommendation labels are accepted")
            if gid not in prior._index:
                outside += 1
                continue
            if label:
                prior.positive[prior._index[gid]] += 1
            else:
                prior.negative[prior._index[gid]] += 1
        count = int(prior.positive.sum() + prior.negative.sum())
        raw_rate = float(prior.positive.sum() / count) if count else 0.5
        # Avoid an improper Beta prior and infinite log odds in all-one-label data.
        prior.global_rate = float(np.clip(raw_rate, 1e-6, 1 - 1e-6))
        prior.metadata = {"training_labels": count, "outside_scope_rows": outside,
                          "unclipped_global_rate": raw_rate,
                          "semantics": "aggregate observed-review approval; not personalized enjoyment probability",
                          "required_split": "population train rows after warm holdout; no validation/test labels"}
        return prior

    def channels(self, ids: Sequence[str], strength: float = 20.0) -> dict[str, np.ndarray]:
        if not math.isfinite(strength) or strength <= 0:
            raise ValueError("Prior strength must be finite and positive")
        positive = np.zeros(len(ids), dtype=np.float64)
        negative = np.zeros(len(ids), dtype=np.float64)
        for out, gid in enumerate(ids):
            index = self._index.get(gid)
            if index is not None:
                positive[out] = self.positive[index]
                negative[out] = self.negative[index]
        count = positive + negative
        approval = (positive + strength * self.global_rate) / (count + strength)
        bounded = np.clip(approval, 1e-12, 1 - 1e-12)
        baseline_logit = math.log(self.global_rate / (1 - self.global_rate))
        return {"approval": approval, "centered_approval": approval - self.global_rate,
                "centered_log_odds": np.log(bounded / (1 - bounded)) - baseline_logit,
                "count": count, "logcount": np.log1p(count),
                "positive_count": positive, "negative_count": negative}

    def score(self, ids: Sequence[str], strength: float = 20.0) -> np.ndarray:
        """Return aggregate posterior approval rates; missing IDs get global prior."""
        return self.channels(ids, strength)["approval"]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        temporary = path / "counts.npz.tmp"
        with temporary.open("wb") as f:
            np.savez_compressed(f, positive=self.positive, negative=self.negative)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path / "counts.npz")
        atomic_json(path / "manifest.json", {"format_version": 1, "game_ids": self.game_ids,
                    "global_rate": self.global_rate, "metadata": self.metadata,
                    "counts_sha256": sha256(path / "counts.npz")})

    @classmethod
    def load(cls, path: Path) -> "QualityPrior":
        path = Path(path)
        try:
            manifest = json.loads((path / "manifest.json").read_text())
            if manifest.get("format_version") != 1:
                raise ValueError("Unsupported quality prior format")
            if sha256(path / "counts.npz") != manifest["counts_sha256"]:
                raise ValueError("Quality prior checksum mismatch")
            prior = cls()
            prior.game_ids = manifest["game_ids"]
            if not isinstance(prior.game_ids, list) or any(not isinstance(gid, str) or not gid for gid in prior.game_ids) or len(set(prior.game_ids)) != len(prior.game_ids):
                raise ValueError("Invalid quality prior IDs")
            prior._index = {gid: i for i, gid in enumerate(prior.game_ids)}
            prior.global_rate = float(manifest["global_rate"])
            if not 0 < prior.global_rate < 1:
                raise ValueError("Invalid quality prior global approval rate")
            with np.load(path / "counts.npz", allow_pickle=False) as data:
                prior.positive = data["positive"].copy()
                prior.negative = data["negative"].copy()
            for counts in (prior.positive, prior.negative):
                if counts.shape != (len(prior.game_ids),) or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
                    raise ValueError("Invalid quality count array")
            prior.metadata = manifest.get("metadata", {})
            return prior
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("Cannot load quality prior; restore or refit the verified artifact") from error


def run_validation(data_dir: Path, report_dir: Path, max_users: int = 300,
                   max_items: int = 1500, seed: int = 42) -> dict:
    """Validation-only grid study. This function never scores held-out test users."""
    import hashlib
    import time
    from .evaluation import cold_queries, prepare, ranking_metrics
    started = time.monotonic()
    data_dir, report_dir = Path(data_dir), Path(report_dir)
    source_hash = sha256(data_dir / "population.sqlite")
    prepared = prepare(data_dir, max_items=max_items, seed=seed)
    version = json.loads((data_dir / "models/current.json").read_text())["version"]
    from .models import Model
    model = Model.load(data_dir / "models" / version)
    if model.metadata.get("source_sha256") != source_hash:
        raise ValueError("Quality source snapshot does not match frozen baseline")
    if prepared["cf_ids"] != model.cf_game_ids:
        raise ValueError("Quality split item IDs do not match baseline")
    prior = QualityPrior.fit(prepared["train_rows"], prepared["cf_ids"])
    prior.metadata.update({"source_sha256": source_hash, "baseline_model_version": version, "baseline_split_sha256": model.metadata.get("split_sha256")})
    artifact = data_dir / "quality/prior"
    prior.save(artifact)
    queries = [q for history in (1, 3, 5) for q in cold_queries(prepared["validation_users"], history, max_users)]
    if not queries:
        raise ValueError("No eligible validation queries")
    ids = prepared["cf_ids"]
    positions = np.asarray([model._idx[gid] for gid in ids])
    canonical = np.asarray(ids)
    cache = []
    for support, seen, targets, negatives in queries:
        candidates = np.flatnonzero([gid not in seen for gid in ids])
        base = np.asarray(model.score(support, model.selected)[positions], dtype=np.float64)
        scale = float(np.max(np.abs(base[candidates]))) if len(candidates) else 0
        if scale:
            base /= scale
        cache.append((base, candidates, targets, negatives))
    trials = []
    record_by_config = {}
    for strength in (5.0, 20.0, 100.0):
        quality = prior.channels(ids, strength)["centered_log_odds"]
        maxabs = float(np.max(np.abs(quality))) if len(quality) else 0
        if maxabs:
            quality /= maxabs
        for weight in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0):
            records = []
            coverage: set[str] = set()
            for base, candidates, targets, negatives in cache:
                scores = (1 - weight) * base + weight * quality
                ranking_indices = candidates[np.lexsort((canonical[candidates], -scores[candidates]))[:20]]
                ranking = canonical[ranking_indices].tolist()
                coverage.update(ranking)
                record = {**ranking_metrics(ranking, targets, 10), **ranking_metrics(ranking, targets, 20)}
                record["explicit_negative_fraction@20"] = len(set(ranking) & negatives) / max(1, len(ranking))
                records.append(record)
            metrics = {key: float(np.mean([r[key] for r in records])) for key in records[0]}
            trial = {"prior_strength": strength, "quality_weight": weight,
                     "kind": "standalone approval" if weight == 1 else "baseline blend",
                     "metrics": metrics, "catalog_coverage@20": len(coverage) / len(ids)}
            trials.append(trial)
            record_by_config[(strength, weight)] = records
    eligible = [trial for trial in trials if trial["quality_weight"] != 1]
    # Stable grid ordering retains the simpler baseline on exact performance ties.
    best = max(eligible, key=lambda trial: trial["metrics"]["ndcg@20"])
    key = (best["prior_strength"], best["quality_weight"])
    selected_records = record_by_config[key]
    baseline_records = record_by_config[(5.0, 0.0)]
    deltas = np.asarray([a["ndcg@20"] - b["ndcg@20"] for a, b in zip(selected_records, baseline_records)])
    rng = np.random.default_rng(seed)
    bootstrap = np.mean(deltas[rng.integers(0, len(deltas), (1000, len(deltas)))], axis=1)
    counts = prior.positive + prior.negative
    source_stats = json.loads((data_dir / "kaggle-population.stats.json").read_text())
    report = {"format_version": 1, "experiment": "validation-only aggregate approval prior",
              "seed": seed, "max_items": max_items, "max_users_per_history": max_users,
              "validation_queries": len(queries), "history_lengths": [1, 3, 5],
              "model_version": version, "baseline_algorithm": model.selected,
              "source_dataset": source_stats["source"], "dataset_sha256": sha256(data_dir / "population.sqlite"),
              "scope": source_stats["selection"], "quality_artifact": str(artifact),
              "training_positive_labels": int(prior.positive.sum()), "training_negative_labels": int(prior.negative.sum()),
              "global_observed_approval": prior.global_rate,
              "train_rows_sha256": hashlib.sha256("\n".join("\t".join(map(str,row)) for row in prepared["train_rows"]).encode()).hexdigest(),
              "count_quantiles": {str(q): float(np.quantile(counts, q)) for q in (0, 0.25, 0.5, 0.75, 0.9, 1)},
              "approval_quantiles_strength20": {str(q): float(np.quantile(prior.score(ids,20), q)) for q in (0,0.25,0.5,0.75,0.9,1)},
              "selected": best, "trials": trials,
              "selected_minus_baseline_ndcg20": {"mean": float(deltas.mean()), "query_bootstrap_ci95": [float(x) for x in np.quantile(bootstrap,[0.025,0.975])]},
              "protocol": "Fit positive/negative counts only from prepare.train_rows after whole-user split and warm holdout. Rank full same 1500 eligible CF IDs excluding support. Baseline is frozen selected population model. Only original validation users scored; no test scoring. Both component channels max-absolute scaled over eligible CF items; quality is centered log odds vs global observed approval. Grid chooses pooled validation NDCG20; standalone approval is diagnostic, not eligible production blend.",
              "limitations": ["Validation selection is optimistic; requires fresh untouched evaluation before claiming an accuracy gain.", "Repeated users across histories make query bootstrap intervals descriptive and not independent-user significance tests.", "Aggregate reviewer approval reflects selection, exposure, era and review participation; not a personalized enjoyment probability.", "Count/logcount are separate notability/exposure channels; observed approval is a different signal.", "Counts cover training-selected 1500-item universe, not the full catalog."],
              "duration_seconds": round(time.monotonic()-started,3)}
    if sha256(data_dir / "population.sqlite") != source_hash:
        raise ValueError("Population snapshot changed during quality validation")
    report_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(report_dir / "quality-prior-results.json", report)
    atomic_json(data_dir / "quality/selected.json", {"prior_strength": best["prior_strength"], "quality_weight": best["quality_weight"], "baseline_algorithm": model.selected, "model_version": version, "source": "original validation users only", "quality_channel": "maxabs centered log odds"})
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report-dir", type=Path, default=Path("docs"))
    parser.add_argument("--max-users", type=int, default=300)
    args = parser.parse_args()
    print(json.dumps(run_validation(args.data_dir,args.report_dir,args.max_users), indent=2))
